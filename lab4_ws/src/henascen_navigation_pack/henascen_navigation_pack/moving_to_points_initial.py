# Inspired by http://wiki.ros.org/turtlesim/Tutorials/Go%20to%20Goal

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Quaternion, Polygon, Point32
from tf_transformations import euler_from_quaternion
import numpy
import matplotlib.pyplot as matpyplot


class movingTurtle(Node):
    def __init__(self):
        super().__init__('turtle_mover')

        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_policy_vel = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=qos_policy
        )

        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile=qos_policy_vel
        )

        self.map_sub = self.create_subscription(
            String,
            'map_path',
            self.map_callback,
            qos_profile=qos_policy
        )

        # self.path_sub = self.create_subscription(
        #     Polygon,
        #     'path_plan',
        #     self.path_callback,
        #     qos_profile=qos_policy
        # )

        self.current_node_pub = self.create_publisher(
            Point32,
            'current_node_pos',
            qos_profile=qos_policy
        )

        self.rob_pose = Odometry()
        self.rob_angles = Quaternion()
        
        self.point_list = (
            numpy.array([
                [0.75, 1.22],   # Node 1
                [1.4, 5.1],     # Node 9
                [3.55, 5.1],    # Node 6
                [3.55, 1.7],    # Node 2
                [5.98, 1.75],   # Node 3
                [6, 5.1],       # Node 6
            ])
        )

        # self.point_list = numpy.array([])

        self.tolerance = 0.1

        self.rate = self.create_rate(2)
        self.timer = self.create_timer(0.5, self.moving_to_point)

        self.current_goal = 0

        self.map_path = None

        self.robot_trajectory = numpy.array([])

        self.grid_size = 200
        self.resolution = 20
        self.grid_center = numpy.array(
            [
                self.grid_size / 4,
                self.grid_size / 4
            ]
        )

        self.robot_pos_xy = numpy.array([])

        self.initial_timer = self.create_timer(
            0.5,
            self.publish_initial_node_pos
        )

        self.angular_velocities = []
        self.linear_velocities = []
        self.iteration = 0
    
    def publish_initial_node_pos(self):

        if self.robot_pos_xy.size > 0 and not self.initial_timer.is_canceled():
            self.publish_current_node_pos(
                self.robot_pos_xy
            )
            self.initial_timer.cancel()
            print('Initial Position Published')

    def odom_callback(self, odom):
        self.rob_pose = odom

        self.rob_pose.pose.pose.position.x = numpy.round(
            self.rob_pose.pose.pose.position.x,
            4
        )
        self.rob_pose.pose.pose.position.y = numpy.round(
            self.rob_pose.pose.pose.position.y,
            4
        )

        self.robot_pos_xy = numpy.array(
            [
                self.rob_pose.pose.pose.position.x,
                self.rob_pose.pose.pose.position.y
            ]
        )

        self.rob_angles = odom.pose.pose.orientation

        try:
            self.robot_trajectory = numpy.vstack(
                (
                    self.robot_trajectory,
                    self.robot_pos_xy
                )
            )
        except:
            self.robot_trajectory = self.robot_pos_xy
    
    def map_callback(self, string):
        string_msg = string

        self.map_path = string_msg.data

    def extract_theta(self):
        orientation_q = self.rob_angles
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ]
        (roll, pitch, yaw) = euler_from_quaternion (
            orientation_list
        )

        return numpy.round(yaw, 4)

    def distance_to_point(self, point_odom):
        """Distance between position and the point to reach"""

        x_square = numpy.square(
            point_odom.pose.pose.position.x - self.rob_pose.pose.pose.position.x
        )
        y_square = numpy.square(
            point_odom.pose.pose.position.y - self.rob_pose.pose.pose.position.y
        )

        return numpy.sqrt(x_square + y_square)
    
    def linear_vel(self, point_odom, constant=0.2):
        """
        Compute the velocity to reach the point as proportional to
        the distance left to reach it
        """
        return constant * self.distance_to_point(point_odom=point_odom)
    
    def steer_angle(self, point_odom):
        """Return the angle to rotate the robot"""
        x_diff = point_odom.pose.pose.position.x - self.rob_pose.pose.pose.position.x
        y_diff = point_odom.pose.pose.position.y - self.rob_pose.pose.pose.position.y

        steering_angle = numpy.arctan2(
            y_diff, x_diff
        )
        print(f'Steering angle {steering_angle}')

        return steering_angle
    
    def error_angle_point(self, steer_angle):
        theta = self.extract_theta()
        # print(f'Robot theta: {theta}')

        error_angle = steer_angle - theta

        if error_angle > numpy.pi:
            steer_dir = -1
        elif error_angle < -numpy.pi:
            steer_dir = -1
        else:
            steer_dir = 1
        
        return steer_dir * error_angle, theta
    
    def angular_vel(self, error_ang_pnt, constant=0.3):
        """Returns the velocity to which the robot will be rotated"""
        # theta = self.extract_theta()

        # return constant * (self.steer_angle(point_odom=point_odom) - theta)

        # error_angle, theta = self.error_angle_point(point_odom)

        return constant * error_ang_pnt
    
    def path_callback(self, path):
        path_positions = path.points

        path_pos_list = []

        for path_pos in path_positions:
            path_pos_x = path_pos.x
            path_pos_y = path_pos.y

            path_pos_list.append(
                numpy.array(
                    [path_pos_x, path_pos_y]
                )
            )
        
        path_pos_list.pop(0)
        self.point_list = numpy.array(path_pos_list)
        self.current_goal = 0
        self.timer = self.create_timer(0.5, self.moving_to_point)
    
    def moving_to_point(self):
        # for point in self.point_list:
        if self.current_goal < len(self.point_list):
            self.iteration += 1
        
            point = self.point_list[self.current_goal]
            print(f"Point Goal: {point}")

            point_to = Odometry()

            point_to.pose.pose.position.x = point[0]
            point_to.pose.pose.position.y = point[1]

            vel_msg = Twist()

            distance = self.distance_to_point(point_odom=point_to)
            steer_angle = self.steer_angle(point_odom=point_to)
            error_angle_dir, theta = self.error_angle_point(steer_angle=steer_angle)
            error_angle = numpy.abs(error_angle_dir)

            if  error_angle > self.tolerance:
                print(f'Angle error: {error_angle}')
                print(f'Angle error dir: {error_angle_dir}')
                print(f'Theta {theta}')
                
                # Angular velocity in the z-axis.
                vel_msg.angular.x = 0.0
                vel_msg.angular.y = 0.0
                vel_msg.angular.z = self.angular_vel(error_ang_pnt=error_angle_dir)
                
                # Publish velocity
                self.angular_velocities.append(
                    numpy.array([self.iteration, vel_msg.angular.z])
                )
                self.vel_publisher.publish(vel_msg)

            elif distance >= self.tolerance:
                # print(f'Robot pos: {self.rob_pose.pose.pose.position}')

                # Linear velocity in the x-axis.
                vel_msg.linear.x = self.linear_vel(point_odom=point_to)
                vel_msg.linear.y = 0.0
                vel_msg.linear.z = 0.0
                
                # Angular velocity in the z-axis.
                # vel_msg.angular.x = 0.0
                # vel_msg.angular.y = 0.0
                # vel_msg.angular.z = self.angular_vel(point_odom=point_to)

                # Publish velocity
                self.linear_velocities.append(
                    numpy.array([self.iteration, vel_msg.linear.x])
                )
                self.vel_publisher.publish(vel_msg)

            else:
                # Stop
                vel_msg.linear.x = 0.0
                vel_msg.angular.z = 0.0
                self.vel_publisher.publish(vel_msg)
                self.current_goal += 1
        
        elif self.current_goal == len(self.point_list):
            # It has finish the trajectory
            print('Trajectory finished')

            self.publish_current_node_pos(
                robot_position=self.robot_pos_xy
            )

            if self.map_path:
                print(self.map_path)
                self.plot_trajectory_in_map(
                    trajectory=self.robot_trajectory,
                    map_path=self.map_path
                )
            
            self.plot_velocities_over_time(
                self.linear_velocities,
                self.angular_velocities
            )
            self.current_goal += 1
            self.timer.destroy()
    
    def trajectory_to_map(self, trajectory, grid_center):
        trajectory_scaled = trajectory * self.resolution
        trajectory_scaled = trajectory_scaled.astype(int)
        new_trajectory_pnt = (
            trajectory_scaled + grid_center
        ).astype(int)

        return new_trajectory_pnt

    def plot_trajectory_in_map(
        self,
        trajectory,
        map_path,
    ):
        map_grid = numpy.genfromtxt(
            map_path,
            dtype='i8',
            delimiter=','
        )

        print(map_grid.shape)

        robot_trajectory = trajectory

        trajectory_coords = self.trajectory_to_map(
            trajectory=robot_trajectory,
            grid_center=self.grid_center
        )

        map_grid[
            trajectory_coords[:, 0],
            trajectory_coords[:, 1]
        ] = 2

        matpyplot.matshow(map_grid)
        matpyplot.savefig("./labs/lab_4/images/moving_mapping_initial.png")
        matpyplot.close()

        print('Trajectory saved in map')
    
    def publish_current_node_pos(self, robot_position):
        current_pos_msg = Point32()

        current_pos_msg.x = numpy.round(robot_position[0], 2)
        current_pos_msg.y = numpy.round(robot_position[1], 2)
        current_pos_msg.z = 0.0

        self.current_node_pub.publish(current_pos_msg)
    
    def plot_velocities_over_time(
        self,
        linear_velocities,
        angular_velocities
    ):
        lineal_plot = numpy.array(linear_velocities)
        angular_plot = numpy.array(angular_velocities)

        fig1, (ax1, ax2) = matpyplot.subplots(2)

        ax1.set_title('Angular Velocities')
        ax1.scatter(angular_plot[:, 0], angular_plot[:, 1])
        # ax1.set_xlim([-1.5, 1.5])
        # ax1.set_ylim([-1, 1])

        ax2.set_title('Lineal Velocities')
        ax2.scatter(lineal_plot[:, 0], lineal_plot[:, 1])
        # ax2.set_xlim([-1.5, 1.5])
        # ax2.set_ylim([-1, 1])

        fig1.tight_layout()
        fig1.savefig('./labs/lab_4/images/twist_messages.png')

def main(args=None):
    rclpy.init(args=args)
    turtle_mover = movingTurtle()

    rclpy.spin(turtle_mover)

    turtle_mover.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
