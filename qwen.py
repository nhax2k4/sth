import gradio as gr
import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import base64
from io import BytesIO
import json

# Set the device to CUDA if available, else fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


model_path = "/app/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

# Cell


def hough_transform_split(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    # Detect horizontal and vertical lines
    kernel_len = gray.shape[1] // 120
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

    image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)

    combined_lines = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
    dilated = cv2.dilate(combined_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cv2.boundingRect(contour) for contour in contours if
                      cv2.boundingRect(contour)[2] > 20 and cv2.boundingRect(contour)[3] > 20]
    valid_contours = sorted(valid_contours, key=lambda bbox: (bbox[1], bbox[0]))

    cropped_images = []
    for x, y, w, h in valid_contours:
        cropped_patch = image[y:y + h, x:x + w]
        cropped_images.append(cropped_patch)

    return cropped_images[1:]

# Function to run inference on each cropped image using a default prompt
def inference(image, prompt="Extract Json"):
    try:
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=200)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Loại bỏ dấu ```json và ``` ở đầu và cuối
        output_text_clean = output_text[0].strip()
        if output_text_clean.startswith("```json"):
            output_text_clean = output_text_clean[len("```json"):].strip()
        if output_text_clean.endswith("```"):
            output_text_clean = output_text_clean[:-len("```")].strip()

        return output_text_clean
    except Exception as e:
        print(f"Error during inference: {e}")
        return None  # Trả về None nếu có lỗi

# Function to create an HTML table to display images and outputs
def generate_html_table(data):
    html = '''
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid black;
        padding: 5px;
        text-align: left;
        vertical-align: top;
    }
    th:first-child, td:first-child {
        width: 50%; /* The first column (left) takes up 50% width */
    }
    img {
        display: block;
        margin: auto;
    }
    </style>
    '''
    html += '<table>'
    html += '<tr><th>Image</th><th>Output</th></tr>'
    for img, output in data:
        # Convert image to base64 to display in HTML
        buffered = BytesIO()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from OpenCV format to PIL
        img.save(buffered, format="PNG")  # Save image to temporary memory in PNG format
        img_str = base64.b64encode(buffered.getvalue()).decode()  # Convert image to base64 string
        img_html = f'<img src="data:image/png;base64,{img_str}"/>'  # Create HTML string to display the image
        html += f'<tr><td>{img_html}</td><td>{output}</td></tr>'  # Add image and result to the table
    html += '</table>'
    return html

# Function to process and display each image after inference with the default prompt
def process_images_one_by_one(image, accumulated_data):
    cropped_images = hough_transform_split(image)

    if accumulated_data is None:
        accumulated_data = []

    json_outputs = []  # List to store JSON objects
    for idx, img in enumerate(cropped_images):
        result = inference(img, "extract text in json format")  # Use default prompt

        if result is None:
            result_clean = ""
        else:
            # Loại bỏ dấu ```json và ``` ở đầu và cuối
            result_clean = result.strip()
            if result_clean.startswith("```json"):
                result_clean = result_clean[len("```json"):].strip()
            if result_clean.endswith("```"):
                result_clean = result_clean[:-len("```")].strip()

        # Append to accumulated data
        accumulated_data.append((img, result_clean))

        # Try to parse the result as JSON
        try:
            json_obj = json.loads(result_clean)
            json_outputs.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error in image {idx+1}: {e}")
            # Skip invalid JSON outputs or add an error object
            json_outputs.append({"error": f"Invalid JSON in image {idx+1}"})

        # Generate HTML to display accumulated data
        accumulated_display_html = generate_html_table(accumulated_data)
        # Update the 'Result' tab incrementally
        yield accumulated_display_html, gr.update()

    # After all images are processed, serialize json_outputs to JSON string
    concatenated_outputs = json.dumps(json_outputs, ensure_ascii=False, indent=4)

    # Update the 'Summary' tab
    yield accumulated_display_html, concatenated_outputs

# Hàm để chuẩn bị nội dung cho việc tải xuống
def prepare_download(concatenated_outputs):
    import base64
    # Không cần cố gắng phân tích cú pháp JSON, chỉ cần lưu chuỗi vào tệp
    json_str = concatenated_outputs
    # Mã hóa chuỗi JSON thành base64
    b64 = base64.b64encode(json_str.encode()).decode()
    # Tạo liên kết tải xuống
    href = f'<a href="data:text/json;base64,{b64}" download="output.json">Nhấn vào đây để tải xuống</a>'
    return href

# Hàm để xóa tất cả các đầu vào và đầu ra
def clear_all():
    return None, [], "", ""

# Tạo giao diện Gradio với ba tab riêng biệt và nút xóa
with gr.Blocks(css=".scroll-box { height: 800px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }") as demo:
    accumulated_data = gr.State([])  # State để lưu trữ kết quả của các hình ảnh đã xử lý

    with gr.Tab("Input"):
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload an Image")

            # Thêm các nút cho tab Input
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear")  # Thêm nút Clear để đặt lại giao diện

    with gr.Tab("Result"):
        # Tab Result sẽ hiển thị bảng kết quả với hộp cuộn lớn hơn (chiều cao 800px)
        accumulated_display = gr.HTML(label="History", elem_classes=["scroll-box"])

    with gr.Tab("Summary"):
        # Tab mới để hiển thị kết quả tổng hợp và nút tải xuống
        concatenated_outputs = gr.Textbox(label="Concatenated Outputs", lines=10)
        download_button = gr.Button("Download")
        download_link = gr.HTML(value="", visible=True)

    # Khi nhấn nút Submit, cập nhật tab History và Summary với dữ liệu mới
    submit_button.click(
        process_images_one_by_one,
        inputs=[input_image, accumulated_data],
        outputs=[accumulated_display, concatenated_outputs]
    )

    # Khi nhấn nút Download, chuẩn bị nội dung để tải xuống
    download_button.click(
        prepare_download,
        inputs=[concatenated_outputs],
        outputs=[download_link]
    )

    # Khi nhấn nút Clear, đặt lại tất cả các trường và dữ liệu tích lũy
    clear_button.click(
        clear_all,
        inputs=[],
        outputs=[input_image, accumulated_data, accumulated_display, concatenated_outputs, download_link]
    )


demo.launch()



