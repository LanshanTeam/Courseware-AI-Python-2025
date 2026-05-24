# Weather Skill 🌤️

基于高德地图 MCP Server 的 Claude Code 天气查询技能。

## 功能特性

- ✅ 查询全国城市实时天气
- ✅ 显示温度、湿度、风力等详细信息
- ✅ 支持自然语言查询
- ✅ 友好的格式化输出

## 安装步骤

1. 克隆或复制本项目到您的工作目录
2. 配置高德地图 API Key
3. 在 Claude Code 中打开项目目录

## 配置

编辑 `.claude/settings.json`，填入您的 API Key：

```json
{
  "mcpServers": {
    "amap": {
      "command": "npx",
      "args": ["-y", "@amap/amap-maps-mcp-server"],
      "env": {
        "AMAP_MAPS_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## 使用方法

在 Claude Code 中直接询问：
- "北京天气怎么样？"
- "查询上海的天气"
- "深圳今天会下雨吗？"

## License

MIT