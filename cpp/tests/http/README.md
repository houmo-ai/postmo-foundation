# JetBrains HTTP Client 测试

本目录中的 `.http` 文件可以在 IntelliJ IDEA、PyCharm 和其他支持 JetBrains HTTP Client 的 IDE 中直接运行。

## 环境

默认环境定义在 `http-client.env.json`：

- `baseUrl`：本地后端地址，默认 `http://127.0.0.1:9912`
- `adminToken`：本地开发管理员 Token，默认 `sk-123`

如果需要测试其他环境，请新建不提交仓库的 `http-client.private.env.json`：

```json
{
  "staging": {
    "baseUrl": "https://staging.example.com",
    "adminToken": "replace-with-real-token"
  }
}
```

该私有文件已经加入仓库根目录的 `.gitignore`。

## 测试文件

- `catalog-products.http`：只读冒烟测试，可安全、反复执行；覆盖健康检查、产品列表、类别过滤和未授权访问。
- `catalog-release-flow.http`：完整写入流程；覆盖产品、版本、Release Note、软件制品、下载地址、组件绑定和发布清单查询。
- `model-catalog-projection.http`：只读验证 Web 模型接口的引擎版本投影及 Catalog 缺失降级行为。

## 执行建议

1. 启动后端，并确认数据库已经执行到最新 Alembic 版本。
2. 先运行 `catalog-products.http` 中的健康检查和产品列表请求。
3. 需要验证完整数据模型时，按顺序运行 `catalog-release-flow.http` 的全部请求。
4. 完整流程使用时间戳生成唯一产品编码，结束时会停用测试产品，因此不会出现在默认的 `active_only=true` 查询结果中。

完整流程有前后依赖，不要从中间开始执行。产品编码和制品 ID 会由响应脚本写入 JetBrains HTTP Client 的全局变量，供后续请求引用。
