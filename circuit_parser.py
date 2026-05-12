"""
电路图PDF解析工具 - 核心解析器
Circuit PDF Parser - Core Parser Engine

Author: Zhang2026-c
Date: 2026-05-12
Version: 0.2.0-alpha
"""

import os
import re
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pdf2image
import easyocr
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font

# Import config
from config import (
    PDF_CONFIG, OCR_CONFIG, IMAGE_CONFIG, COMPONENT_CONFIG,
    OUTPUT_CONFIG, PATHS_CONFIG, LOGGING_CONFIG, ADVANCED_CONFIG
)

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['LEVEL']),
    format=LOGGING_CONFIG['FORMAT'],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGGING_CONFIG['FILE'])
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class Component:
    """元器件数据类"""
    name: str              # 元器件名称 (R1, C2等)
    type: str              # 元器件类型 (R, C等)
    page: int              # 出现页码
    confidence: float      # 识别置信度
    location: Tuple       # 在图像中的位置 (x, y)
    raw_text: str         # 原始识别文本

@dataclass
class ParseResult:
    """解析结果数据类"""
    components: List[Component]
    total_processed: int
    total_pages: int
    average_confidence: float
    accuracy_rate: float
    processing_time: float
    errors: List[str]

# ============================================================================
# Circuit PDF Parser
# ============================================================================
class CircuitPDFParser:
    """
    电路图PDF解析器
    
    支持功能：
    - PDF转图像
    - OCR文本识别
    - 元器件自动识别
    - 结果导出（Excel、CSV、JSON）
    """
    
    def __init__(self):
        """初始化解析器"""
        self.logger = logger
        self._setup_directories()
        self._init_ocr()
        self.components_patterns = COMPONENT_CONFIG['PATTERNS']
        
    def _setup_directories(self):
        """创建必需的目录"""
        for dir_name in [PATHS_CONFIG['INPUT_DIR'], 
                         PATHS_CONFIG['OUTPUT_DIR'],
                         PATHS_CONFIG['TEMP_DIR'],
                         PATHS_CONFIG['IMAGES_DIR']]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory ready: {dir_name}")
    
    def _init_ocr(self):
        """初始化OCR引擎"""
        try:
            self.logger.info("Initializing OCR engine...")
            self.ocr_reader = easyocr.Reader(
                OCR_CONFIG['LANGUAGES'],
                gpu=OCR_CONFIG['GPU']
            )
            self.logger.info("OCR engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {e}")
            raise
    
    # ========================================================================
    # PDF Processing
    # ========================================================================
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        将PDF转换为图像列表
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            图像列表
        """
        try:
            self.logger.info(f"Converting PDF to images: {pdf_path}")
            
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=PDF_CONFIG['DPI'],
                fmt=PDF_CONFIG['FORMAT']
            )
            
            # 限制页数
            if PDF_CONFIG['MAX_PAGES']:
                images = images[PDF_CONFIG['START_PAGE']:PDF_CONFIG['START_PAGE'] + PDF_CONFIG['MAX_PAGES']]
            
            self.logger.info(f"Successfully converted {len(images)} pages")
            return images
            
        except Exception as e:
            self.logger.error(f"Error converting PDF: {e}")
            raise
    
    # ========================================================================
    # Image Processing
    # ========================================================================
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: PIL Image对象
            
        Returns:
            预处理后的OpenCV图像
        """
        # PIL转OpenCV
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 灰度化
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 对比度增强（CLAHE）
        if IMAGE_CONFIG['CONTRAST_ENHANCE']:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # 二值化
        _, binary = cv2.threshold(gray, IMAGE_CONFIG['THRESHOLD'], 255, cv2.THRESH_BINARY)
        
        # 降噪
        if IMAGE_CONFIG['DENOISE']:
            binary = cv2.medianBlur(binary, IMAGE_CONFIG['DENOISE_STRENGTH'])
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (IMAGE_CONFIG['MORPH_KERNEL'], IMAGE_CONFIG['MORPH_KERNEL']))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        self.logger.debug("Image preprocessing completed")
        return binary
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        增强图像质量
        
        Args:
            image: PIL Image对象
            
        Returns:
            增强后的图像
        """
        # 增强对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # 增强锐度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    # ========================================================================
    # OCR and Component Recognition
    # ========================================================================
    def extract_text_with_ocr(self, image_array: np.ndarray) -> List[Tuple]:
        """
        使用OCR提取文本
        
        Args:
            image_array: OpenCV图像数组
            
        Returns:
            OCR识别结果列表 [(bbox, text, confidence), ...]
        """
        try:
            results = self.ocr_reader.readtext(image_array)
            self.logger.debug(f"OCR extracted {len(results)} text blocks")
            return results
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return []
    
    def identify_components(self, ocr_results: List[Tuple]) -> List[Component]:
        """
        识别OCR结果中的电子元器件
        
        Args:
            ocr_results: OCR识别结果
            
        Returns:
            识别到的Component列表
        """
        components = []
        
        for (bbox, text, confidence) in ocr_results:
            # 过滤低置信度结果
            if confidence < OCR_CONFIG['CONFIDENCE_THRESHOLD']:
                continue
            
            # 清理文本
            cleaned_text = self._clean_text(text)
            
            # 识别元器件类型
            component_type = self._match_component_type(cleaned_text)
            
            if component_type:
                # 获取位置
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                
                component = Component(
                    name=cleaned_text,
                    type=component_type,
                    page=0,  # 页码在调用时设置
                    confidence=confidence,
                    location=(x, y),
                    raw_text=text
                )
                components.append(component)
                self.logger.debug(f"Identified component: {cleaned_text} (type: {component_type}, conf: {confidence:.2f})")
        
        return components
    
    def _clean_text(self, text: str) -> str:
        """
        清理识别文本
        
        Args:
            text: 原始识别文本
            
        Returns:
            清理后的文本
        """
        # 移除空格
        text = text.replace(' ', '')
        # 转换为大写
        text = text.upper()
        # 移除特殊字符
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
    
    def _match_component_type(self, text: str) -> Optional[str]:
        """
        匹配文本对应的元器件类型
        
        Args:
            text: 清理后的文本
            
        Returns:
            元器件类型或None
        """
        for comp_type, pattern in self.components_patterns.items():
            if re.match(pattern, text):
                return comp_type
        return None
    
    # ========================================================================
    # Main Parsing
    # ========================================================================
    def parse_pdf(self, pdf_path: str) -> ParseResult:
        """
        解析PDF文件并提取元器件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            ParseResult对象
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting PDF parsing: {pdf_path}")
        
        try:
            # 转换PDF为图像
            images = self.convert_pdf_to_images(pdf_path)
            total_pages = len(images)
            
            # 解析每一页
            all_components = []
            errors = []
            
            for page_num, image in enumerate(images):
                try:
                    self.logger.info(f"Processing page {page_num + 1}/{total_pages}")
                    
                    # 图像增强
                    enhanced_image = self.enhance_image(image)
                    
                    # 图像预处理
                    processed_image = self.preprocess_image(enhanced_image)
                    
                    # OCR识别
                    ocr_results = self.extract_text_with_ocr(processed_image)
                    
                    # 元器件识别
                    components = self.identify_components(ocr_results)
                    
                    # 设置页码
                    for component in components:
                        component.page = page_num + 1
                    
                    all_components.extend(components)
                    
                except Exception as e:
                    error_msg = f"Error processing page {page_num + 1}: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # 计算统计信息
            processing_time = time.time() - start_time
            average_confidence = np.mean([c.confidence for c in all_components]) if all_components else 0
            accuracy_rate = len(all_components) / (len(all_components) + len(errors)) * 100 if (all_components or errors) else 0
            
            result = ParseResult(
                components=all_components,
                total_processed=len(all_components),
                total_pages=total_pages,
                average_confidence=average_confidence,
                accuracy_rate=accuracy_rate,
                processing_time=processing_time,
                errors=errors
            )
            
            self.logger.info(f"PDF parsing completed in {processing_time:.2f}s")
            self.logger.info(f"Found {len(all_components)} components with {average_confidence:.2%} avg confidence")
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDF parsing failed: {e}")
            raise
    
    # ========================================================================
    # Results Processing
    # ========================================================================
    def print_summary(self, result: ParseResult):
        """
        打印结果摘要
        
        Args:
            result: ParseResult对象
        """
        print("\n" + "="*70)
        print("电路图PDF解析结果摘要 (Circuit PDF Parse Summary)")
        print("="*70)
        print(f"总页数: {result.total_pages}")
        print(f"识别元器件数: {result.total_processed}")
        print(f"平均置信度: {result.average_confidence:.2%}")
        print(f"识别准确率: {result.accuracy_rate:.2%}")
        print(f"处理时间: {result.processing_time:.2f}s")
        print(f"错误数: {len(result.errors)}")
        
        if result.components:
            print("\n元器件类型统计:")
            type_count = defaultdict(int)
            for component in result.components:
                type_count[component.type] += 1
            
            for comp_type in sorted(type_count.keys()):
                print(f"  {comp_type}: {type_count[comp_type]}")
        
        print("="*70 + "\n")
    
    def save_results(self, result: ParseResult, output_file: Optional[str] = None) -> str:
        """
        保存结果到文件
        
        Args:
            result: ParseResult对象
            output_file: 输出文件路径（可选）
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(PATHS_CONFIG['OUTPUT_DIR'], f"circuit_parse_{timestamp}.xlsx")
        
        try:
            self.logger.info(f"Saving results to: {output_file}")
            self._save_to_excel(result, output_file)
            self.logger.info(f"Results saved successfully")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def _save_to_excel(self, result: ParseResult, output_file: str):
        """
        保存结果到Excel文件
        
        Args:
            result: ParseResult对象
            output_file: 输出文件路径
        """
        wb = Workbook()
        ws_main = wb.active
        ws_main.title = "Components"
        
        # 写入组件数据
        headers = ["元器件", "类型", "页码", "置信度"]
        ws_main.append(headers)
        
        # 设置表头样式
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        for cell in ws_main[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # 写入数据
        for component in sorted(result.components, key=lambda x: (x.page, x.name)):
            ws_main.append([
                component.name,
                component.type,
                component.page,
                f"{component.confidence:.2%}"
            ])
        
        # 统计工作表
        ws_stats = wb.create_sheet("Statistics")
        stats_data = [
            ["指标", "值"],
            ["总页数", result.total_pages],
            ["识别元器件数", result.total_processed],
            ["平均置信度", f"{result.average_confidence:.2%}"],
            ["识别准确率", f"{result.accuracy_rate:.2%}"],
            ["处理时间(秒)", f"{result.processing_time:.2f}"],
            ["错误数", len(result.errors)],
        ]
        
        for row in stats_data:
            ws_stats.append(row)
        
        # 按类型分类
        type_count = defaultdict(list)
        for component in result.components:
            type_count[component.type].append(component)
        
        for comp_type, components in sorted(type_count.items()):
            ws_type = wb.create_sheet(f"Type_{comp_type}")
            ws_type.append(["元器件", "页码", "置信度"])
            
            for component in sorted(components, key=lambda x: (x.page, x.name)):
                ws_type.append([
                    component.name,
                    component.page,
                    f"{component.confidence:.2%}"
                ])
        
        # 调整列宽
        for ws in wb.sheetnames:
            ws_obj = wb[ws]
            for column in ws_obj.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                ws_obj.column_dimensions[column_letter].width = min(max_length + 2, 50)
        
        # 保存
        wb.save(output_file)

# ============================================================================
# Utility Functions
# ============================================================================
def parse_pdf_file(pdf_path: str) -> ParseResult:
    """
    快速解析PDF文件
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        ParseResult对象
    """
    parser = CircuitPDFParser()
    result = parser.parse_pdf(pdf_path)
    return result

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # 处理pdfs目录中的所有PDF文件
    pdf_dir = Path(PATHS_CONFIG['INPUT_DIR'])
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(pdf_files)} PDF file(s)")
    
    parser = CircuitPDFParser()
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file}")
        
        try:
            result = parser.parse_pdf(str(pdf_file))
            parser.print_summary(result)
            parser.save_results(result)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")
