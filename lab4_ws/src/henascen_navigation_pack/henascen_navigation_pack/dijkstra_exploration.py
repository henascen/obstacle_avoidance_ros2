from pathlib import Path
from queue import PriorityQueue

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32, Polygon
import numpy
import networkx
import matplotlib.pyplot as matpyplot

MAIN_DIR = Path.cwd()

class DijkstraOnce(Node):
    def __init__(self):
        super().__init__('dijkstra_once_planner')

        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.path_publisher = self.create_publisher(
            Polygon,
            'path_plan',
            qos_profile=qos_policy
        )

        self.current_pos_sub = self.create_subscription(
            Point32,
            'current_node_pos',
            self.current_pos_callback,
            qos_profile=qos_policy
        )

        self.point_list = [
                (0.0, 0.0),         # Node 0
                (0.75, 1.22),   # Node 1
                (3.55, 1.7),    # Node 2
                (5.98, 1.75),   # Node 3
                (6.00, 3.12),   # Node 4
                (6.00, 5.1),    # Node 5
                (3.55, 5.1),    # Node 6
                (3.55, 3.6),    # Node 7
                (1.4, 3.6),     # Node 8
                (1.4, 5.1),     # Node 9
            ]

        self.images_path = MAIN_DIR / 'labs' / 'lab_4' / 'images'

        graph_h, positions, adj_matrix = self.build_adjacency_matrix()

        self.graph_h = graph_h
        self.positions = positions
        self.adj_matrix = adj_matrix

        self.visited = set()

        # self.create_timer(1, self.publish_planning)
    
    def current_pos_callback(self, current_pos):
        current_x = numpy.round(current_pos.x, 2)
        current_y = numpy.round(current_pos.y, 2)

        print(f'Current robot position: {current_x, current_y}')

        current_node = None

        for node, position in self.positions.items():
            diff_x = numpy.round(
                numpy.abs(current_x - position[0]),
                1
            )
            diff_y = numpy.round(
                numpy.abs(current_y - position[1]),
                1
            )

            if diff_x <= 0.1 and diff_y <= 0.1:
                current_node = node
                print(f'Current robot node: {current_node}')
                break

        if current_node != None:
            print('Computing plan path')
            self.publish_planning(
                initial_node=current_node
            )
    
    def create_nodes(self, graph, point_list):
        for node_number, node_position in enumerate(point_list):
            graph.add_node(node_number, pos=node_position)


    def building_neighbours(self):
        """This is created manually, """
        neigh_pairs = [
            (0, 1), (1, 2), (1, 8), (2, 1), (2, 7), (2, 3),
            (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6),
            (6, 5), (6, 7), (6, 9), (7, 2), (7, 6), (8, 9),
            (8, 1), (9, 8), (1, 0), (9, 6)
        ]

        return neigh_pairs
    
    def compute_distance(self, coords_a, coords_b):
        va = numpy.array(coords_a)
        vb = numpy.array(coords_b)

        distance_rnd = numpy.round(
            numpy.linalg.norm(va - vb),
            4
        )

        return distance_rnd

    def add_distance_to_neighbours(
        self,
        graph,
        neigh_pairs
    ):
        neigh_pairs_wdist = []

        positions = networkx.get_node_attributes(graph, 'pos')

        for pair in neigh_pairs:
            point_a = pair[0]
            point_b = pair[1]

            pos_a = positions[point_a]
            pos_b = positions[point_b]

            distance_ab = self.compute_distance(
                coords_a=pos_a,
                coords_b=pos_b
            )

            neigh_pairs_wdist.append(
                (point_a, point_b, distance_ab)
            )
        
        return neigh_pairs_wdist, positions

    def build_graph(self):
        graph_h = networkx.Graph()

        self.create_nodes(
            graph=graph_h,
            point_list=self.point_list
        )

        neigh_pairs = self.building_neighbours()

        neigh_pairs, pos = self.add_distance_to_neighbours(
            graph=graph_h,
            neigh_pairs=neigh_pairs
        )

        graph_h.add_weighted_edges_from(
            neigh_pairs
        )
        networkx.draw(
            graph_h,
            pos=pos
        )

        labels = networkx.get_edge_attributes(graph_h, 'weight')
        networkx.draw_networkx_edge_labels(
            graph_h,
            pos,
            edge_labels=labels
        )
        matpyplot.savefig(str(self.images_path) + '/graph_draw.png')
        print('Graph created and saved succesfully')

        return graph_h, pos
    
    def create_adjacency_matrix(self, graph):
        n_nodes = graph.number_of_nodes()

        adj_matrix = numpy.zeros([n_nodes, n_nodes])

        for node_number in range(0, n_nodes):
            neighbours = [
                neighbor
                for neighbor in graph.neighbors(node_number)
            ]

            adj_matrix[node_number, neighbours] = 1
        
        return adj_matrix
    
    def build_adjacency_matrix(self):
        graph_h, positions = self.build_graph()
        adj_matrix = self.create_adjacency_matrix(
            graph=graph_h
        )

        matpyplot.matshow(adj_matrix)
        matpyplot.savefig(str(self.images_path) + '/adjmatrix_draw.png')
        print('Adjacency Matrix created succesfully')

        return graph_h, positions, adj_matrix
    
    def run_dijkstra_path_plan(
        self,
        initial_node,
        graph,
        adjacency_matrix
    ):
        start_node = initial_node

        # Store the nodes we have visited
        # visited = set()

        # Store the paths with distances as weights
        pqueue = PriorityQueue()
        
        # We add the initial node, where we begin
        # Distance is zero
        # We store each element in the queue as:
        #   (distance, node_to_go, complete_path_to_node_to_go)
        pqueue.put((0, start_node, [start_node]))

        # We add the inital node to visited because we have visited
        print(start_node)
        self.visited.add(start_node)

        while not pqueue.empty():

            # Get the next value in the queue according to distances
            current_cost, current_node, current_path = (
                pqueue.get()
            )

            print(self.visited)

            # If we are in the final node then stop
            if current_node not in self.visited:
                print(f'We have next node: {current_node}')
                print(f'The path is {current_path}')
                # current_path = []
                break
            
            # We get the neighbor flags for all the nodes for the current node
            node_neighbors_adjmat = adjacency_matrix[current_node]

            for neighbor_n, is_neighbor in enumerate(node_neighbors_adjmat):
                if is_neighbor:
                    
                    # if neighbor_n not in self.visited:
                        # visited.add(neighbor_n)

                    distance_cost = graph.get_edge_data(
                        current_node,
                        neighbor_n
                    )['weight']

                    pqueue.put(
                        (
                            current_cost + distance_cost,
                            neighbor_n,
                            current_path + [neighbor_n]
                        )
                    )

                        # break

        return current_path
    
    def build_path_positions_list(self, positions, path_plan):
        
        positions_list = [
            positions[node_plan] for node_plan in path_plan
        ]

        return positions_list
    
    def path_planning(
        self,
        graph,
        adj_matrix,
        positions,
        initial_node=0,
    ):

        path_plan = self.run_dijkstra_path_plan(
            initial_node=initial_node,
            graph=graph,
            adjacency_matrix=adj_matrix
        )

        # print(path_plan)

        path_positions = self.build_path_positions_list(
            positions=positions,
            path_plan=path_plan
        )

        print(path_positions)

        return path_positions

    def publish_planning(self, initial_node=0):
        
        path_positions = self.path_planning(
            graph=self.graph_h,
            adj_matrix=self.adj_matrix,
            positions=self.positions,
            initial_node=initial_node
        )

        positions_msg = Polygon()
        positions_msg_list = []

        for path_position in path_positions:
            position_msg = Point32()

            position_msg.x = (path_position[0])
            position_msg.y = path_position[1]
            position_msg.z = 0.0

            positions_msg_list.append(position_msg)
        
        positions_msg.points = positions_msg_list

        self.path_publisher.publish(positions_msg)
        print('Path planning published')
    
    def current_node_pos(self, position):
        pass


def main(args=None):

    rclpy.init(args=args)
    dijkstra_once = DijkstraOnce()

    rclpy.spin(dijkstra_once)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
