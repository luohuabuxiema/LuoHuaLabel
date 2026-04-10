# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
import queue
from PySide6.QtCore import QObject, QThread, Signal
from PIL import Image

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class ModelLoadWorker(QThread):
    loaded = Signal(object, object, bool, str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            model = build_sam3_image_model(checkpoint_path=self.checkpoint_path, enable_inst_interactivity=True)
            model.to("cuda")
            processor = Sam3Processor(model)
            self.loaded.emit(model, processor, True, "模型加载成功")
        except Exception as e:
            self.loaded.emit(None, None, False, str(e))


class SamInferenceWorker(QThread):
    # 🟢 修复：增加了第五个 list 参数，用于传输 OBB 旋转框数据
    result_ready = Signal(list, list, list, float, bool)

    text_result_ready = Signal(list, str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.inference_state = None
        self.task_queue = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            try:
                task_type, data, is_click = self.task_queue.get(timeout=0.05)

                if not self.model or not self.inference_state:
                    continue

                if task_type == 'point':
                    x, y = data
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = self.model.predict_inst(
                            inference_state=self.inference_state,
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=is_click
                        )

                    if len(scores) > 0:
                        best_idx = np.argmax(scores)
                        mask_np = masks[best_idx].cpu().numpy() if torch.is_tensor(masks) else masks[best_idx]
                        score_val = float(scores[best_idx].cpu() if torch.is_tensor(scores) else scores[best_idx])

                        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        poly_pts = []
                        rect_xywh = []
                        rect_obb = []  # 🟢 新增 OBB 容器
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                            poly_pts = approx.reshape(-1, 2).tolist()

                            x_r, y_r, w_r, h_r = cv2.boundingRect(largest_contour)
                            rect_xywh = [x_r, y_r, w_r, h_r]

                            # 🟢 计算最小外接旋转矩形 OBB
                            obb = cv2.minAreaRect(largest_contour)
                            rect_obb = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2]]

                        # 🟢 修复：发送 5 个参数
                        self.result_ready.emit(poly_pts, rect_xywh, rect_obb, score_val, is_click)

                # ================= 文本提示词智能提取分支 =================
                elif task_type == 'text':
                    prompt_text = data
                    if not self.processor:
                        continue

                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out_state = self.processor.set_text_prompt(prompt=prompt_text, state=self.inference_state)

                        masks = out_state.get("masks", [])
                        scores = out_state.get("scores", [])
                        boxes = out_state.get("boxes", [])

                        results = []
                        if len(masks) > 0:
                            for i in range(len(masks)):
                                mask_np = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                                mask_np = np.squeeze(mask_np)

                                score_val = float(scores[i].cpu() if torch.is_tensor(scores[i]) else scores[i])
                                box = boxes[i].cpu().numpy() if torch.is_tensor(boxes[i]) else boxes[i]

                                if box.ndim > 1:
                                    box = box.squeeze()
                                x1, y1, x2, y2 = box
                                rect_xywh = [x1, y1, x2 - x1, y2 - y1]

                                mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                poly_pts = []
                                rect_obb = []
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                                    poly_pts = approx.reshape(-1, 2).tolist()

                                    # 🟢 计算最小外接旋转矩形 OBB
                                    obb = cv2.minAreaRect(largest_contour)
                                    rect_obb = [obb[0][0], obb[0][1], obb[1][0], obb[1][1], obb[2]]

                                if poly_pts:
                                    results.append({
                                        "poly_pts": poly_pts,
                                        "rect": rect_xywh,
                                        "obb": rect_obb,  # 🟢 装入字典
                                        "score": score_val
                                    })

                        self.text_result_ready.emit(results, prompt_text)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"推理错误: {e}")

    def request_inference(self, x, y, is_click=False):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                pass
        self.task_queue.put(('point', (x, y), is_click))

    def request_text_inference(self, prompt_text):
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                pass
        self.task_queue.put(('text', prompt_text, True))

    def stop(self):
        self.running = False
        self.wait()


class SAMClient(QObject):
    model_status_changed = Signal(bool, str)
    # 🟢 修复：增加了第五个 list 参数
    inference_result = Signal(list, list, list, float, bool)
    text_result_ready = Signal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.processor = None
        self.inference_worker = SamInferenceWorker()
        self.inference_worker.result_ready.connect(self.inference_result)
        self.inference_worker.text_result_ready.connect(self.text_result_ready)
        self.inference_worker.start()
        self.load_worker = None

    def load_model_async(self, checkpoint_path):
        self.model_status_changed.emit(False, "正在后台加载模型，请稍候...")
        self.load_worker = ModelLoadWorker(checkpoint_path)
        self.load_worker.loaded.connect(self._on_model_loaded)
        self.load_worker.start()

    def _on_model_loaded(self, model, processor, success, msg):
        if success:
            self.model = model
            self.processor = processor
            self.inference_worker.model = model
            self.inference_worker.processor = processor
        self.model_status_changed.emit(success, msg)

    def set_image(self, image_path):
        if not self.processor: return
        try:
            pil_img = Image.open(image_path).convert("RGB")
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = self.processor.set_image(pil_img)
                self.inference_worker.inference_state = state
        except Exception as e:
            print(f"图像特征提取失败: {e}")

    def request_inference(self, x, y, is_click):
        if self.model:
            self.inference_worker.request_inference(x, y, is_click)

    def request_text_inference(self, prompt_text):
        if self.model:
            self.inference_worker.request_text_inference(prompt_text)

    def cleanup(self):
        self.inference_worker.stop()