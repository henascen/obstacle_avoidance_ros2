from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion
import numpy
import matplotlib.pyplot as matpyplot

import bresenham_points

MAIN_DIR = Path.cwd()

class WorldMapper(Node):
    def __init__(self):
        super().__init__('lidar_mapping')
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile=qos_policy
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=qos_policy
        )

        self.map_path_pub = self.create_publisher(
            String,
            '/map_path',
            qos_profile=qos_policy
        )

        self.odom_robot_position = numpy.array([])
        self.odom_robot_orientation = numpy.array([])

        self.grid_size = 200
        self.resolution = 20

        self.center = numpy.array(
            [
                self.grid_size / 4,
                self.grid_size / 4
            ]
        )

        self.previous_center = self.center.astype(int)

        self.robot_odom = Odometry()
        self.robot_pos_xy = numpy.array([0, 0])
        self.robot_theta = 0.0

        self.robot_scan = LaserScan()
        self.robot_ranges_angles = numpy.array([])

        self.timer = self.create_timer(0.25, self.mapping)
        self.map_timer = self.create_timer(
            5,
            self.save_numpy_map
        )

        self.grid_map = self.create_grid_map(
            grid_size=self.grid_size
        )

        self.robot_trajectory = numpy.array([])

        self.map_save_path = MAIN_DIR / 'labs' / 'lab_4' / 'map' / 'henmap_initial.csv'
    
    def extract_robot_xy_theta(self, odom_msg):
        """Return array with x, y and also yaw in the tuple"""
        robot_x = numpy.round(
            odom_msg.pose.pose.position.x,
            4
        )
        robot_y = numpy.round(
            odom_msg.pose.pose.position.y,
            4
        )
        robot_pos_xy = numpy.array(
            [robot_x, robot_y]
        )

        quaternion = odom_msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion (
            [
                quaternion.x,
                quaternion.y,
                quaternion.z,
                quaternion.w
            ]
        )

        theta = numpy.round(yaw, 4)

        return robot_pos_xy, theta
        
    
    def odom_callback(self, odom):
        self.robot_odom = odom

        robot_xy, theta = self.extract_robot_xy_theta(
            odom_msg=self.robot_odom
        )

        self.robot_pos_xy = robot_xy
        self.robot_theta = theta
    
    def extract_ranges_angles(self, scan_msg):
        """Return ranges, angles in a numpy array - remove inf ranges"""
        initial_angle = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        ranges = numpy.array(scan_msg.ranges)

        angles = numpy.arange(
            start=initial_angle,
            stop=( (2 * numpy.pi) ),
            step=angle_increment
        )

        polar_coord = numpy.column_stack(
            (ranges, angles)
        )

        polar_coord = polar_coord[
            ~numpy.isinf(polar_coord).any(axis=1)
        ]

        return polar_coord
    
    def scan_callback(self, scan):
        self.robot_scan = scan

        self.robot_ranges_angles = self.extract_ranges_angles(
            scan_msg=scan
        )
    
    def rotate_points(self, data, theta):
        x_ = (
            data[:,0] * numpy.cos(theta) - data[:,1] * numpy.sin(theta)
        )
        y_=  (
            data[:,0] * numpy.sin(theta) + data[:,1] * numpy.cos(theta)
        )
        
        cart_coord = numpy.column_stack(
            (x_, y_)
        )
        return cart_coord

    def convert_polar_to_cartes(self, ranges_angles, theta):
        ranges = ranges_angles[:, 0]
        angles = ranges_angles[:, 1]

        xs = ranges * numpy.cos(angles)
        ys = ranges * numpy.sin(angles)

        xs_l = [xs_ for xs_ in xs if (xs_ > -1000 and xs_ < 1000)]
        ys_l = [ys_ for ys_ in ys if (ys_ > -1000 and ys_ < 1000)]

        cart_coord = numpy.column_stack(
            (xs_l, ys_l)
        )

        cart_coord = self.rotate_points(
                data=cart_coord,
                theta=theta
        )
        
        return cart_coord
    
    def plot_raw_data(
        self,
        ranges_angles,
        coords_xy
    ):
        fig1 = matpyplot.figure()
        ax1 = matpyplot.subplot(121)
        ax2 = matpyplot.subplot(122, projection='polar')
        
        ax2.plot(
            ranges_angles[:, 1],
            ranges_angles[:, 0],
            '.'
        )

        ax1.plot(
            coords_xy[:, 0],
            coords_xy[:, 1],
            '.'
        )
        
        fig1.tight_layout()
        fig1.savefig('./labs/lab_4/images/raw_data/raw_data_polarCarte_initial.png')

    def create_grid_map(self, grid_size):
        # grid_map = numpy.zeros(
        #     [grid_size, grid_size]
        # )

        grid_map = numpy.full(
            (grid_size, grid_size),
            -1,
            dtype=int
        )

        return grid_map
    
    def scale_carte_coord_grid_size(
        self,
        carte_coord,
        robot_pos,
        center
    ):
        robot_pos_scaled = robot_pos * self.resolution
        robot_pos_scaled = robot_pos_scaled.astype(int)
        new_center = (robot_pos_scaled + center).astype(int)

        carte_coord = carte_coord * self.resolution
        carte_coord = carte_coord.astype(int)
        new_carte_coord = (carte_coord + new_center).astype(int)

        return new_carte_coord, new_center
    
    def build_grid_map(
        self,
        grid_map,
        carte_coord,
        robot_pos,
        center
    ):

        mapping_coords, robot_pos = (
            self.scale_carte_coord_grid_size(
                carte_coord=carte_coord,
                robot_pos=robot_pos,
                center=center
            )
        )

        for x_scaled, y_scaled in mapping_coords:
            empty_points = bresenham_points.bresenham_points(
                robot_pos,
                [x_scaled, y_scaled]
            )
            empty_points = numpy.asarray(empty_points)

            grid_map[
                empty_points[:, 0],
                empty_points[:, 1]
            ] = 0

        # Record the trajectory of the robot
        # grid_map[
        #     self.previous_center[0],
        #     self.previous_center[1]
        # ] = 2
        
        # Put all the others coords as occupied
        grid_map[
            mapping_coords[:, 0],
            mapping_coords[:, 1]
        ] = 1
        
        # As well as the center (occupied)
        # grid_map[robot_pos[0], robot_pos[1]] = 1
        
        # Save the current center to be the previous one
        self.previous_center = robot_pos
        self.robot_trajectory = numpy.append(
            arr=self.robot_trajectory,
            values=self.previous_center,
            axis=0
        )

        return grid_map

    def mapping(self):
        if self.robot_ranges_angles.size > 0:
            ranges_angles = self.robot_ranges_angles
            theta = self.robot_theta
            robot_pos_xy = self.robot_pos_xy

            coords_xy = self.convert_polar_to_cartes(
                ranges_angles=ranges_angles,
                theta=theta
            )

            # self.plot_raw_data(
            #     ranges_angles=ranges_angles,
            #     coords_xy=coords_xy
            # )

            grid_map = self.build_grid_map(
                grid_map=self.grid_map,
                carte_coord=coords_xy,
                robot_pos=robot_pos_xy,
                center=self.center
            )

            self.grid_map = grid_map
    
    def save_numpy_map(self):
        grid_map = self.grid_map
        
        matpyplot.matshow(grid_map)
        matpyplot.savefig("./labs/lab_4/images/lidar_mapping_initial.png")
        matpyplot.close()

        map_path = str(self.map_save_path)

        grid_map_int = grid_map.astype(int)

        numpy.savetxt(
            map_path,
            grid_map_int,
            fmt='%i',
            delimiter=',',
        )

        self.publish_string(
            string_to_pub=map_path,
            publisher=self.map_path_pub
        )
    
    def publish_string(self, string_to_pub, publisher):
        string_msg = String()
        string_msg.data = string_to_pub

        publisher.publish(
            string_msg
        )

def main(args=None):

    rclpy.init(args=args)
    lidar_mapper = WorldMapper()

    rclpy.spin(lidar_mapper)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
