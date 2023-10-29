# SILL: Semantic Integrated LiDAR Labelling

## Dependencies
* VisPy
* ROS
* NumPy
* SciPy
* Open3D
* [semantics_manager](https://github.com/KumarRobotics/semantics_manager)

## Usage

```
rosrun sill sill.py [--start N] [--num_panos P] [--output_dir D] [--config C] [--load] path_to_dataset.bag
```
The `start` flag selects how many panos should be skipped to start labelling.
The `num_panos` flag selects how many panos should be loaded at a time (defaults to 10).
The `config` flag selects the location of the config file (the default is `config/config.yaml`), which contains more settings.
The `output_dir` flag manually sets the location of the output directory, overriding the default.
Setting `load` will cause the tool to look for existing labels, and if they exist will prelabel the cloud.

The dataset bag should include the following topics and can be created by recording these topics while running [ouster_decoder](https://github.com/KumarRobotics/ouster_decoder) and [rofl](https://github.com/versatran01/rofl-beta).
```
/os_node/camera_info                : sensor_msgs/CameraInfo
/os_node/img                        : sensor_msgs/Image
/os_node/rofl_odom/pano/camera_info : sensor_msgs/CameraInfo
/os_node/rofl_odom/pano/image       : sensor_msgs/Image
/os_node/rofl_odom/sweep/cloud      : sensor_msgs/PointCloud2
```

The upper right corner shows the current class being labelled, the z-axis cutoff, and the index of the last shown sweep.
To load `P` more sweeps into the current map to label, press `n`.
SILL works by slicing the world in z
All points above the current elevation, shown in the upper left, are hidden.
When clicking and dragging, all points inside the circle below the selected z will be labelled the current class, but points below that have been labelled previously will not be overwritten.
To change z, Use `PgUp`/`PgDown` and press `R` to redraw at the selected height.
When `W` is pressed, labels will be written to disk in a folder in the same place as the bag file.

## Hotkey reference
- `Space`: When held, pan when clicking and dragging
- `Scroll Wheel`: Zoom
- `N`: Load `P` more sweeps
- `PgUp`/`PgDown`: Adjust the current elevation by 0.5m increments.  Note that the scene will not be redrawn
- `R`: Rerender the scene
- `O`: Enter 3D orbit mode.  Cannot label in this mode.
- `L`: Enter top-down labelling mode.
- `W`: Write labels to disk.
- `X`: Clear the current display.
- `0`-`9`: Select class
