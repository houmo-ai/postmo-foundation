# hliepython

多进程、多实例的 OpenAI 兼容本地推理服务。

详细说明见 [hliepython/README.md](hliepython/README.md)。

## 打包

建议在已准备好的 Python 3.11 环境中执行：

```bash
python -m pip install build
python -m build --wheel
```

产物默认位于 `dist/` 目录。

## 安装

对于已经预装好 `torch`、`torchaudio`、`vllm` 以及硬件相关依赖的环境，建议使用：

```bash
python -m pip install --no-deps dist/hliepython-0.1.1-py3-none-any.whl
```

然后按实际启用的能力补齐依赖。

## 环境要求

- Python 3.11 推荐
- 文本/ASR 依赖 `vllm`
- TTS 依赖 `tcim_lite` 以及相应模型运行环境
- 若使用项目中的 Houmo 定制能力，还需要可导入的 `vllm_houmo`