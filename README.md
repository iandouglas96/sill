# SILL: Semantic Integrated LiDAR Labelling

## Dependencies
* VisPy
* ROS
* NumPy
* SciPy

## Usage

```
rosrun sill sill.py [--start N] [--period T] [--load] path_to_dataset.bag
```
The `period` flag selects how many seconds should elapse between selected sweeps to add.
The `start` flag selets how many sweeps should be skipped to start labelling.
Note that changing `period` will affect the `start` point, since each of the `N` panos are spaced by `T`.
Setting `load` will cause the tool to look for existing labels, and if they exist will prelabel the cloud.

The dataset bag should include the following topics and can be created by recording these topics while running [ouster_decoder](https://github.com/KumarRobotics/ouster_decoder) and [llol](https://github.com/versatran01/llol).
```
/os_node/camera_info      : sensor_msgs/CameraInfo 
/os_node/image            : sensor_msgs/Image      
/os_node/llol_odom/sweep  : sensor_msgs/PointCloud2
/tf                       : tf2_msgs/TFMessage
```

The upper right corner shows the current class being labelled, the z-axis cutoff, and the index of the last shown sweep.
To load 10 more sweeps into the current map to label, press `n`.
SILL works by slicing the world in z
All points above the current elevation, shown in the upper left, are hidden.
When clicking and dragging, all points inside the circle below the selected z will be labelled the current class, but points below that have been labelled previously will not be overwritten.
To change z, Use `PgUp`/`PgDown` and press `R` to redraw at the selected height.
When `W` is pressed, labels will be written to disk in a folder in the same place as the bag file.

## Hotkey reference
- `Space`: When held, pan when clicking and dragging
- `Scroll Wheel`: Zoom
- `N`: Load 10 more sweeps
- `PgUp`/`PgDown`: Adjust the current elevation by 0.5m increments.  Note that the scene will not be redrawn
- `R`: Rerender the scene
- `O`: Enter 3D orbit mode.  Cannot label in this mode.
- `L`: Enter top-down labelling mode.
- `W`: Write labels to disk.
- `X`: Clear the current display.
- `0`-`9`: Select class
