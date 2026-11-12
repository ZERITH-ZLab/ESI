# 评测快照与恢复

每次新评测会在输出目录保存源码快照和配置，供所有 worker 及后续重试使用。

```text
<output_root>/evaluation/<name>/
  source_snapshot/
  queue_config.json
  queue/
  retries/
```

恢复时使用相同实验名、配置和输出目录。已完成 case 会保留，未完成 episode 重新开始。
程序校验快照、权重和环境身份；身份不一致时会停止，避免混用不同策略的结果。

修改策略、模型或任务配置后，请使用新的实验名。更新工作区不会替换已有实验的策略快照。
不要删除校验文件绕过检查；快照损坏或丢失时，应恢复原始备份或新建实验。

启动方法见[评测教程](EVALUATION.md)。
