import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import numpy

class PotentialPath(Node):

    GRAD_STEP_SIZE = 0.1

    def __init__(self):
        super().__init__('potential_path')

        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # self.current_pos_sub = self.create_subscription(
        #     Point32,
        #     'current_node_pos',
        #     self.current_pos_callback,
        #     qos_profile=qos_policy
        # )

        # self.final_goal_sub = self.create_subscription(
        #     Point32,
        #     'final_node_pos',
        #     self.final_pos_callback,
        #     qos_profile=qos_policy
        # )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=qos_policy
        )

        self.lidar_sub  = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile=qos_policy
        )

        self.robot_pos_xy = numpy.array([])
        self.robot_theta = numpy.array([])

        # Testing
        self.final_goal = numpy.array([5, 0])

        self.current_point = numpy.array([0, 0])

        # The obstacle position is the reading + the robot position
        self.obstacle_point = numpy.array([])
    
    def odom_callback(self, odom):
        self.robot_odom = odom

        robot_xy, theta = self.extract_robot_xy_theta(
            odom_msg=self.robot_odom
        )

        self.robot_pos_xy = robot_xy
        self.robot_theta = theta

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

    def lidar_callback(self, lidar_msg):
        initial_angle = lidar_msg.angle_min
        angle_increment = lidar_msg.angle_increment

        ranges = numpy.array(lidar_msg.ranges)

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

        if self.robot_pos_xy.size > 0 and self.robot_theta.size > 0:
            
            self.obstacle_point = self.get_obstacle_point(
                polar_coord=polar_coord,
                theta=self.robot_theta,
                robot_pos_xy=self.robot_pos_xy
            )

            self.grad_descent_potential(
                current_point=self.robot_pos_xy,
                goal_point=self.final_goal,
                obstacle_point=self.obstacle_point
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
    
    def get_obstacle_point(
        self,
        polar_coord,
        theta,
        robot_pos_xy
    ):

        if polar_coord.size > 0:
            cart_coord = self.convert_polar_to_cartes(
                ranges_angles=polar_coord,
                theta=theta
            )
            obstacle_point = cart_coord.mean(axis=0) + robot_pos_xy
        else:
            obstacle_point = None

        return obstacle_point

    def point_total_gradient_potential(
        self,
        gradient_att_potential,
        gradient_rep_potential
    ):
        return gradient_att_potential + gradient_rep_potential

    def point_gradient_att_potential(
        self,
        current_point,      # narray as [x, y]
        goal_point,         # narray as [x, y]
        d_thres_goal=0.5,
        chi=1,
    ):
        d_curr_goal = numpy.linalg.norm(current_point - goal_point)
        if d_curr_goal <= d_thres_goal:
            grad_att = chi * (current_point - goal_point)
        else:
            grad_att = (
                (d_thres_goal * chi * (current_point - goal_point)) /
                d_curr_goal
            )

        return grad_att

    def point_gradient_rep_potential(
        self,
        current_point,
        obstacle_point,
        d_safe_thres=1,
        eta=1
    ):
        d_curr_obst = numpy.linalg.norm(current_point - obstacle_point)
        if d_curr_obst <= d_safe_thres:
            comp_1 = ( (1 / d_safe_thres) - (1 / (d_curr_obst)) )
            comp_2 = (1 / (numpy.square(d_curr_obst)))
            comp_3 = ( 
                (current_point - obstacle_point) /
                d_curr_obst
            )

            grad_rep = eta * comp_1 * comp_2 * comp_3
        else:
            grad_rep = 0
        
        return grad_rep

    def grad_descent_potential(
        self,
        current_point,
        goal_point,
        obstacle_point,
    ):
        goal_error = numpy.linalg.norm(current_point - goal_point)
        while goal_error > 0.1:
            
            grad_att_pot = self.point_gradient_att_potential(
                current_point=current_point,
                goal_point=goal_point
            )
            grad_rep_pot = self.point_gradient_rep_potential(
                current_point=current_point,
                obstacle_point=obstacle_point
            )
            total_grad_pot = self.point_total_gradient_potential(
                gradient_att_potential=grad_att_pot,
                gradient_rep_potential=grad_rep_pot
            )
            
            next_point = (
                current_point - (self.GRAD_STEP_SIZE * total_grad_pot)
            )
            print(f'Next point to go: {next_point}')

            current_point = next_point
            goal_error = numpy.linalg.norm(current_point - goal_point)

def main(args=None):

    rclpy.init(args=args)
    potential_path = PotentialPath()

    rclpy.spin(potential_path)

    rclpy.shutdown()

if __name__ == '__main__':
    main()