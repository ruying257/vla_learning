import mujoco
import mujoco.viewer

xml_path = "/home/p/vla/mode/scene.xml"

# 加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 启动查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 保持查看器运行，直到用户关闭窗口
    while viewer.is_running():
        # 执行一步仿真
        mujoco.mj_step(model, data)
        # 更新查看器中的场景
        viewer.sync()
