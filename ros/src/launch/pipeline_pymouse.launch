<launch>
    <node name="JoystickNode" pkg="assistive_controller" type="spacemouse_input_node.py" output="screen" />
    <!-- <node name="SystemSimulationNode" pkg="assistive_controller" type="simulation_node.py" output="screen" />  -->
    <node name="VisulaizationNode" pkg="assistive_controller" type="visualization_node.py" output="screen" /> 
    <node name="FrankaNode" pkg="assistive_controller" type="franka_controller_node.py" output="screen" />
    
    <arg name="Assistance_Active" default="1"/>
    <arg name="Training_Active" default="0"/>
    <arg name="restart_sim" default="0"/>
    <param name="ass_active" type="bool" value="$(arg Assistance_Active)"/>
    <param name="train_active" type="bool" value="$(arg Training_Active)"/>
    <param name="restart_sim" type="bool" value="$(arg restart_sim)"/>
    
    <node pkg="assistive_controller" type="assistive_control_node.py" name="AssistiveControlNode1" output="screen"/> 
    <!-- Get the data path from ROS parameter server -->
    <!-- Conditional rosbag record node -->
</launch>
