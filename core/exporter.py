import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from core.shapes import RectShape, PolyShape, PointShape, RotatedRectShape


class Exporter:
    @staticmethod
    def extract_shapes(scene):
        shapes_data = []
        for item in scene.items():
            if isinstance(item, RectShape) and not getattr(item, 'is_temp', False):
                rect = item.rect()
                p1 = item.mapToScene(rect.topLeft())
                p2 = item.mapToScene(rect.bottomRight())
                shapes_data.append({
                    "label": item.label,
                    "type": "rectangle",
                    "points": [[p1.x(), p1.y()], [p2.x(), p2.y()]]
                })
            elif isinstance(item, PolyShape) and not getattr(item, 'is_temp', False):
                poly = item.polygon()
                mapped_poly = item.mapToScene(poly)
                points = [[pt.x(), pt.y()] for pt in mapped_poly]
                shapes_data.append({
                    "label": item.label,
                    "type": "polygon",
                    "points": points
                })
            elif isinstance(item, PointShape) and not getattr(item, 'is_temp', False):
                rect = item.rect()
                shapes_data.append({
                    "label": item.label,
                    "type": "point",
                    "points": [[rect.center().x(), rect.center().y()]]
                })
            elif isinstance(item, RotatedRectShape) and not getattr(item, 'is_temp', False):
                poly = item.polygon()
                points = [[poly[i].x(), poly[i].y()] for i in range(4)]

                cx = item.pos().x()
                cy = item.pos().y()
                w = item.box_w
                h = item.box_h
                angle = item.rotation()

                shapes_data.append({
                    "label": item.label,
                    "type": "obb",
                    "points": points,
                    "rect": [cx, cy, w, h],  # JSON 中保存: [中心X, 中心Y, 宽, 高]
                    "angle": angle
                })
        shapes_data.reverse()
        return shapes_data

    @staticmethod
    def save_json(filepath, image_path, image_width, image_height, shapes):
        data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageHeight": int(image_height),
            "imageWidth": int(image_width)
        }
        for s in shapes:
            shape_type = s["type"]
            shape_dict = {
                "label": s["label"],
                "points": s["points"],
                "group_id": None,
                "shape_type": shape_type,
                "flags": {}
            }
            if shape_type == "obb":
                shape_dict["angle"] = s.get("angle", 0)
                shape_dict["rect"] = s.get("rect", [0, 0, 0, 0])

            data["shapes"].append(shape_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_yolo(filepath, image_width, image_height, shapes, classes_list):
        lines = []
        for s in shapes:
            if s["label"] not in classes_list:
                continue
            class_id = classes_list.index(s["label"])

            if s["type"] == "rectangle":
                x1, y1 = s["points"][0]
                x2, y2 = s["points"][1]
                cx = ((x1 + x2) / 2.0) / image_width
                cy = ((y1 + y2) / 2.0) / image_height
                w = abs(x2 - x1) / image_width
                h = abs(y2 - y1) / image_height
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            elif s["type"] == "obb":
                flat_pts = []
                for pt in s["points"][:4]:
                    flat_pts.append(f"{pt[0] / image_width:.6f} {pt[1] / image_height:.6f}")
                lines.append(f"{class_id} " + " ".join(flat_pts))

            elif s["type"] == "polygon":
                flat_pts = []
                for pt in s["points"]:
                    flat_pts.append(f"{pt[0] / image_width:.6f} {pt[1] / image_height:.6f}")
                lines.append(f"{class_id} " + " ".join(flat_pts))

            elif s["type"] == "point":
                cx = s["points"][0][0] / image_width
                cy = s["points"][0][1] / image_height
                pw, ph = 0.02, 0.02  # 微小框宽高
                cx = max(pw / 2, min(1.0 - pw / 2, cx))
                cy = max(ph / 2, min(1.0 - ph / 2, cy))
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {pw:.6f} {ph:.6f}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    @staticmethod
    def save_xml(filepath, image_path, image_width, image_height, shapes):
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = os.path.basename(image_path)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(int(image_width))
        ET.SubElement(size, "height").text = str(int(image_height))
        ET.SubElement(size, "depth").text = "3"

        for s in shapes:
            min_x = min(p[0] for p in s["points"])
            max_x = max(p[0] for p in s["points"])
            min_y = min(p[1] for p in s["points"])
            max_y = max(p[1] for p in s["points"])

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = s["label"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(min_x))
            ET.SubElement(bndbox, "ymin").text = str(int(min_y))
            ET.SubElement(bndbox, "xmax").text = str(int(max_x))
            ET.SubElement(bndbox, "ymax").text = str(int(max_y))

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(xmlstr)