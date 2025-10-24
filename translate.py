#!/usr/bin/python3
import requests
import json
import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI


# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feedln.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("translate")

class LLMTranslator:
    """LLM大模型翻译器类"""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        初始化翻译器
        
        Args:
            model_config: 模型配置字典，包含API密钥、模型名称等
        """
        # 默认配置
        self.default_config = {
            "api_key": os.environ.get("LLM_API_KEY", ""),
            "model_name": "gpt-3.5-turbo",  # 默认使用OpenAI的模型
            "api_base": "https://api.openai.com/v1/chat/completions",
            "timeout": 30,
            "max_tokens": 1000
        }
        
        # 更新为用户提供的配置
        self.config = {**self.default_config, **(model_config or {})}
        
        # 验证必要的配置
        if not self.config["api_key"]:
            logger.warning("未配置LLM API密钥，请设置LLM_API_KEY环境变量或在配置中提供")
    
    def _prepare_request(self, text: str) -> Dict[str, Any]:
        """准备翻译请求"""
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的翻译助手，请将用户提供的英文文本翻译成中文，保持原意准确，语言流畅自然。"
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        request_data = {
            "model": self.config["model_name"],
            "messages": messages,
            "max_tokens": self.config["max_tokens"]
        }
        
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        return {
            "url": self.config["api_base"],
            "headers": headers,
            "json": request_data,
            "timeout": self.config["timeout"]
        }
    
    def translate0(self, text: str) -> str:
        """
        执行翻译
        
        Args:
            text: 要翻译的英文文本
            
        Returns:
            翻译后的中文文本
        """
        if not text.strip():
            return ""
        
        if not self.config["api_key"]:
            raise ValueError("LLM API密钥未配置，请设置LLM_API_KEY环境变量或在配置中提供")
        
        try:
            # 准备请求数据
            request_kwargs = self._prepare_request(text)
            
            # 发送请求
            #logger.info(f"正在调用LLM模型进行翻译，文本长度: {len(text)}字符")
            response = requests.post(**request_kwargs)
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            translated_text = result["choices"][0]["message"]["content"]
            
            #logger.info(f"翻译完成，结果长度: {len(translated_text)}字符")
            return translated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"翻译请求失败: {str(e)}")
            raise Exception(f"翻译请求失败: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"解析翻译结果失败: {str(e)}")
            raise Exception(f"解析翻译结果失败: {str(e)}")
        except Exception as e:
            logger.error(f"翻译过程中发生未知错误: {str(e)}")
            raise

    def  translate(self, text: str) -> str:
        """
        执行翻译
        
        Args:
            text: 要翻译的英文文本
            
        Returns:
            翻译后的中文文本
        """

        # 构造 client
        client = OpenAI(
            api_key=self.config["api_key"],  # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
        )
        completion = client.chat.completions.create(
            model=self.config["model_name"],
            messages=[
                {
                    "role": "user",
                    "content": f"请将英文翻译成中文, 只保留译文，不要输出其他内容: {text}"
                }
            ],
            extra_body={
                "enable_enhancement": True,  # <- 自定义参数
            },
        )
        #print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def summarize(self, text: str) -> str:
        """
        生成文本的中文摘要
        
        Args:
            text: 要摘要的英文文本
            
        Returns:
            中文摘要文本
        """
        # 构造 client
        client = OpenAI(
            api_key=self.config["api_key"],  # 混元 APIKey
            base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
        )
        completion = client.chat.completions.create(
            model=self.config["model_name"],
            messages=[
                {
                    "role": "user",
                    "content": f"请为下文生成一段150字左右的中文摘要，只保留摘要内容，不要输出其他内容: {text}"
                }
            ],
            extra_body={
                "enable_enhancement": True,  # <- 自定义参数
            },
        )
        #print(completion.choices[0].message.content)
        return completion.choices[0].message.content

# 全局翻译器实例
_translator = None

def englishTranslate(text: str, model_config: Optional[Dict[str, Any]] = None) -> str:
    """
    将英文文本翻译为中文
    
    Args:
        text: 要翻译的英文文本
        model_config: 可选的模型配置字典，用于覆盖默认配置
        
    Returns:
        翻译后的中文文本
    """
    global _translator
    
    # 如果没有翻译器实例或提供了新的配置，则创建新实例
    if _translator is None or model_config:
        _translator = LLMTranslator(model_config)
    
    try:
        return _translator.translate(text)
    except Exception as e:
        logger.error(f"翻译失败: {str(e)}")
        # 在发生错误时返回原文，确保程序不会中断
        return text
    
def englishSummarize(text: str, model_config: Optional[Dict[str, Any]] = None) -> str:
    """
    将英文文本生成中文摘要
    
    Args:
        text: 要摘要的英文文本
        model_config: 可选的模型配置字典，用于覆盖默认配置
        
    Returns:
        中文摘要文本
    """
    global _translator
    
    # 如果没有翻译器实例或提供了新的配置，则创建新实例
    if _translator is None or model_config:
        _translator = LLMTranslator(model_config)
    
    try:
        summary_result = _translator.summarize(text)
        summary_result.replace("\n", "")
        return summary_result
    except Exception as e:
        logger.error(f"摘要失败: {str(e)}")
        # 在发生错误时返回原文，确保程序不会中断
        return text
    

if __name__ == "__main__":
    # 示例用法
    try:
        custom_config = {
            "api_key": "sk-XX",  # 可以直接在这里提供API密钥
            "model_name": "hunyuan-turbos-latest",  # 使用不同的模型
            "api_base": "https://api.hunyuan.cloud.tencent.com/v1/chat/completions",  # 可以修改为其他兼容接口
            "timeout": 60,  # 调整超时时间
            "max_tokens": 5000  # 调整最大token数
        }
        # 测试翻译
        test_text = "The rapid development of multimodal large language models(MLLMs) raises the question of how they compare to human performance."
        result = englishTranslate(test_text, custom_config)
        print(f"英文: {test_text}")
        print(f"中文: {result}")
        
        # 测试摘要
        summary_result = englishSummarize(test_text, custom_config)
        print(f"摘要: {summary_result}")
        
    except Exception as e:
        print(f"错误: {str(e)}")