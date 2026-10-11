# v2.1.1 岛屿开发入口

几何、forcing 帮助库和广覆盖配置与 TOY_SERVER 同步：圆岛/椭圆、九类单峰海况。
矩阵与波谱输入不扩展，默认有效训练目标为 16,384。

`build_island_toy.py` 生成开发用地形/forcing 库、图片和稀疏候选索引：

```bash
python build_island_toy.py --overwrite
```

它准备的是候选计划，不代表已经获得有效训练标签。旧开发 runner/converter 保留，
本版本的有效配额、替补运行和严格导出生产流程统一使用 [TOY_SERVER](../TOY_SERVER/README.md)。
具体地形定义、参数、target/reserve、静态检查状态见 [V2_1_1.md](../V2_1_1.md)。
本次没有执行生成、SWAN 或训练。
