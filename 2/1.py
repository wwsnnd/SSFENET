# vit_model/test_import_transformer.py
import sys
import os

# 添加 vit_model 的上级路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vit_model.vit_transformer import Transformer
    print("✅ 成功导入 vit_model.transformer.Transformer")
except ImportError as e:
    print("❌ 无法导入 vit_model.transformer.Transformer")
    print("错误信息:", e)

# 额外打印路径帮助定位问题
import vit_model.transformer
print("模块实际位置:", vit_model.transformer.__file__)
