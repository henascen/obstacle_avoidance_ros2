import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, Polygon
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

        self.current_pos_sub = self.create_subscription(
            Point32,
            'current_node_pos',
            self.current_pos_callback,
            qos_profile=qos_policy
        )

        self.global_path_sub = self.create_subscription(
            Polygon,
            'path_plan',
            self.path_plan_callback,
            qos_profile=qos_policy
        )

        self.next_pos_pub = self.create_publisher(
            Point32,
            'next_pos_move',
            qos_profile=qos_policy
        )

        self.current_goal_pub = self.create_publisher(
            Point32,
            'current_goal_pos',
            qos_profile=qos_policy
        )

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

        self.global_path = numpy.array([])
        self.goal_to_go = 0
    
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

            # If we get a valid lidar measurement, we send a command to
            # later activate the obstacle avoidance
            # if self.obstacle_point.size > 0:
            #     self.publish_position(
            #         publisher=self.next_pos_pub,
            #         robot_position=self.robot_pos_xy
            #     )

            # self.grad_descent_potential(
            #     current_point=self.robot_pos_xy,
            #     goal_point=self.final_goal,
            #     obstacle_point=self.obstacle_point
            # )
    
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
            obstacle_point = numpy.array([])

        return obstacle_point

    def current_pos_callback(self, current_pos_point):
        current_x = numpy.round(current_pos_point.x, 2)
        current_y = numpy.round(current_pos_point.y, 2)

        print(f'Current robot position: {current_x, current_y}')

        current_position = numpy.array(
                [current_x, current_y]
        )
        obstacle_point = self.obstacle_point
        print(f"Obstacle point {obstacle_point}")

        # When receiving the current pos it means that the robot reached
        # the desired position, thus we have to calculate the next position
        # and share it with the robot

        if self.global_path.size > 0:
            # If there's a global plan we start moving
            
            if self.goal_to_go > ( len(self.global_path) - 1 ):
                print('We have completed our plan - notify Dijkstra')
                # We notify Dijkstra that we completed the plan
                # and wait for the next one
                self.publish_position(
                    publisher=self.current_goal_pub,
                    robot_position=current_position
                )
            else:

                current_goal = self.global_path[self.goal_to_go]

                goal_error = numpy.abs(
                    numpy.linalg.norm(current_goal - current_position)
                )

                if goal_error < 0.15:
                    # Publishing the same position, just to get this loop again
                    print("Goal reached, publishing move to the current position")
                    self.publish_position(
                        publisher=self.next_pos_pub,
                        robot_position=current_position
                    )
                    print('We can move to the next goal in the plan')
                    self.goal_to_go += 1
                else:

                    if self.obstacle_point.size > 0:
                        next_pos_distance = 0
                        grad_current_pos = current_position

                        while next_pos_distance < 0.15:
                            print('Gradient Descent - Until next significant position')
                            next_position = self.grad_descent_potential(
                                current_point=grad_current_pos,
                                goal_point=current_goal,
                                obstacle_point=obstacle_point
                            )

                            next_pos_distance = numpy.linalg.norm(
                                current_position - next_position
                            )
                            if next_pos_distance < 0.15:
                                grad_current_pos = next_position

                    else:
                        next_position = current_goal

                    # We publish the next position and wait for this callback to be called again
                    self.publish_position(
                        publisher=self.next_pos_pub,
                        robot_position=next_position
                    )
                    print(f'Next position published: {next_position}')
        else:
            # If there's no plan then the next position to go is the current position
            print(f'Publishing same position as next - no plan {current_position}')
            self.publish_position(
                publisher=self.next_pos_pub,
                robot_position=current_position
            )
            # We also publish to dijkstra to say that we have reached the initial
            # position so that can also plan the next one
            self.publish_position(
                    publisher=self.current_goal_pub,
                    robot_position=current_position
                )

    def publish_position(self, publisher, robot_position):
        current_pos_msg = Point32()

        current_pos_msg.x = numpy.round(robot_position[0], 2)
        current_pos_msg.y = numpy.round(robot_position[1], 2)
        current_pos_msg.z = 0.0

        publisher.publish(current_pos_msg)

    def path_plan_callback(self, path_plan):
        path_positions = path_plan.points

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
        self.global_path = numpy.array(path_pos_list)
        print(f'Path received from dijkstra: {self.global_path}')
        self.goal_to_go = 0

        # Moving the robot to its current position so that the
        # moving node can restart the movement
        self.publish_position(
            publisher=self.next_pos_pub,
            robot_position=numpy.round(self.robot_pos_xy, 2)
        )


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
        d_thres_goal=0.2,
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
        # goal_error = numpy.linalg.norm(current_point - goal_point)
        # while goal_error > 0.1:
            
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

            # current_point = next_point
            # goal_error = numpy.linalg.norm(current_point - goal_point)
        
        return next_point

def main(args=None):

    rclpy.init(args=args)
    potential_path = PotentialPath()

    rclpy.spin(potential_path)

    rclpy.shutdown()

if __name__ == '__main__':
    main()