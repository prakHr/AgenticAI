from typing import Callable, Dict, List, Optional, Tuple


class MiniGraph:
    """A tiny reproduction of LangGraph's StateGraph mechanics."""

    def __init__(self):
        self.nodes: Dict[str, Callable[[dict], dict]] = {}

        # Fixed edges:
        # {
        #     "node_a": "node_b"
        # }
        self.edges: Dict[str, str] = {}

        # Conditional edges:
        # {
        #     "node_a": router_function
        # }
        self.cond_edges: Dict[str, Callable[[dict], str]] = {}

        # Possible destinations of conditional edges:
        # {
        #     "node_a": ["node_b", "node_c"]
        # }
        #
        # This is required because a Python router function
        # cannot always be inspected to determine all possible
        # destinations automatically.
        self.cond_routes: Dict[str, List[str]] = {}

        self.entry: Optional[str] = None

        self.END = "__END__"

    # ---------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------

    def add_node(
        self,
        name: str,
        fn: Callable[[dict], dict]
    ):
        self.nodes[name] = fn
        return self

    def set_entry(self, name: str):
        if name not in self.nodes:
            raise ValueError(
                f"Entry node '{name}' has not been added."
            )

        self.entry = name
        return self

    def add_edge(self, src: str, dst: str):
        self.edges[src] = dst
        return self

    def add_conditional_edge(
        self,
        src: str,
        router: Callable[[dict], str],
        destinations: List[str]
    ):
        """
        Add a conditional edge.

        Parameters
        ----------
        src:
            Source node.

        router:
            Function that receives the current state and
            returns the next node.

        destinations:
            All possible nodes that the router can return.

        Example
        -------
        graph.add_conditional_edge(
            "B",
            lambda state: "C" if state["x"] else "D",
            ["C", "D"]
        )
        """

        self.cond_edges[src] = router
        self.cond_routes[src] = destinations

        return self

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------

    def run(
        self,
        state: dict,
        max_steps: int = 3,
        trace: bool = True
    ) -> dict:

        if self.entry is None:
            raise ValueError(
                "Graph entry node has not been set."
            )

        node = self.entry
        steps = 0

        # Find the longest possible path in the graph.
        path, length = self.longest_root_to_leaf_path()

        # Scale max_steps using the longest path.
        max_steps = int(max_steps * length)

        if trace:
            print(
                f"Longest path: {' -> '.join(path)}"
            )

            print(
                f"Longest path length: {length}"
            )

            print(
                f"Maximum execution steps: {max_steps}"
            )

        while node != self.END and steps < max_steps:

            if trace:
                print(
                    f"[step {steps}] -> node: {node}"
                )

            if node not in self.nodes:
                raise ValueError(
                    f"Node '{node}' does not exist."
                )

            # Execute node
            fn = self.nodes[node]

            state = fn(state)

            # -------------------------------------------------
            # Conditional edge has priority
            # -------------------------------------------------

            if node in self.cond_edges:

                node = self.cond_edges[node](state)

            # -------------------------------------------------
            # Fixed edge
            # -------------------------------------------------

            elif node in self.edges:

                node = self.edges[node]

            # -------------------------------------------------
            # No outgoing edge -> END
            # -------------------------------------------------

            else:

                node = self.END

            steps += 1

        if trace:
            print(
                f"[step {steps}] -> END"
            )

        state["_steps"] = steps

        return state

    # ---------------------------------------------------------
    # Build adjacency list
    # ---------------------------------------------------------

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """
        Build an adjacency list containing both:

        1. Fixed edges
        2. Conditional edge destinations

        Example:

            A -> B
            B -> C
            B -> D

        becomes:

            {
                "A": ["B"],
                "B": ["C", "D"],
                "C": [],
                "D": []
            }
        """

        adjacency = {
            node: []
            for node in self.nodes
        }

        # -----------------------------------------------------
        # Fixed edges
        # -----------------------------------------------------

        for src, dst in self.edges.items():

            if src not in adjacency:
                adjacency[src] = []

            adjacency[src].append(dst)

        # -----------------------------------------------------
        # Conditional edges
        # -----------------------------------------------------

        for src, destinations in self.cond_routes.items():

            if src not in adjacency:
                adjacency[src] = []

            for dst in destinations:

                if dst not in adjacency[src]:
                    adjacency[src].append(dst)

        return adjacency

    # ---------------------------------------------------------
    # Longest root -> leaf path
    # ---------------------------------------------------------

    def longest_root_to_leaf_path(
        self,
        include_end: bool = False
    ) -> Tuple[List[str], int]:
        """
        Find the longest possible path from the entry/root
        node to a leaf.

        Both fixed and conditional edges are considered.

        Returns
        -------
        (path, length)

        Example:

            A -> B -> C -> D

        returns:

            (
                ["A", "B", "C", "D"],
                4
            )

        Length represents the number of nodes.
        """

        if self.entry is None:
            raise ValueError(
                "Graph entry node has not been set."
            )

        adjacency = self._build_adjacency()

        # DFS memoization
        memo: Dict[str, Tuple[List[str], int]] = {}

        # Nodes currently being visited.
        # Used for cycle detection.
        visiting = set()

        def dfs(node: str):

            # -------------------------------------------------
            # Cycle detection
            # -------------------------------------------------

            if node in visiting:

                raise ValueError(
                    f"Cycle detected while finding longest "
                    f"path: {node}"
                )

            # -------------------------------------------------
            # Already calculated
            # -------------------------------------------------

            if node in memo:
                return memo[node]

            visiting.add(node)

            # -------------------------------------------------
            # Leaf node
            # -------------------------------------------------

            if node not in adjacency or not adjacency[node]:

                visiting.remove(node)

                result = (
                    [node],
                    1
                )

                memo[node] = result

                return result

            # -------------------------------------------------
            # Find longest child path
            # -------------------------------------------------

            longest_path = []
            longest_length = 0

            for child in adjacency[node]:

                child_path, child_length = dfs(child)

                current_length = 1 + child_length

                if current_length > longest_length:

                    longest_length = current_length

                    longest_path = [
                        node
                    ] + child_path

            visiting.remove(node)

            result = (
                longest_path,
                longest_length
            )

            memo[node] = result

            return result

        path, length = dfs(self.entry)

        # -----------------------------------------------------
        # Remove END from displayed path
        # -----------------------------------------------------

        if not include_end and self.END in path:

            path = path[:path.index(self.END)]

            length = len(path)

        return path, length

    # ---------------------------------------------------------
    # Pretty print longest path
    # ---------------------------------------------------------

    def print_longest_root_to_leaf_path(self):

        path, length = (
            self.longest_root_to_leaf_path()
        )

        print(
            "\nLongest Root -> Leaf Path"
        )

        print(
            "-" * 40
        )

        print(
            " -> ".join(path)
        )

        print(
            f"\nNumber of nodes : {length}"
        )

        print(
            f"Number of edges : {max(length - 1, 0)}"
        )

        return path, length


# =============================================================
# Example
# =============================================================

if __name__ == "__main__":

    graph = MiniGraph()

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    graph.add_node(
        "A",
        lambda state: {
            **state,
            "value": state.get("value", 0) + 1
        }
    )

    graph.add_node(
        "B",
        lambda state: {
            **state,
            "value": state["value"] + 1
        }
    )

    graph.add_node(
        "C",
        lambda state: {
            **state,
            "value": state["value"] + 1
        }
    )

    graph.add_node(
        "D",
        lambda state: {
            **state,
            "value": state["value"] + 1
        }
    )

    graph.add_node(
        "E",
        lambda state: {
            **state,
            "value": state["value"] + 1
        }
    )

    graph.add_node(
        "F",
        lambda state: {
            **state,
            "value": state["value"] + 1
        }
    )

    # ---------------------------------------------------------
    # Entry
    # ---------------------------------------------------------

    graph.set_entry("A")

    # ---------------------------------------------------------
    # Fixed edge
    #
    # A -> B
    # ---------------------------------------------------------

    graph.add_edge(
        "A",
        "B"
    )

    # ---------------------------------------------------------
    # Conditional edge
    #
    # B -> C
    # B -> E
    # ---------------------------------------------------------

    graph.add_conditional_edge(
        "B",
        lambda state: (
            "C"
            if state["value"] % 2 == 0
            else "E"
        ),
        [
            "C",
            "E"
        ]
    )

    # ---------------------------------------------------------
    # Remaining fixed edges
    #
    # C -> D
    # E -> F
    # ---------------------------------------------------------

    graph.add_edge(
        "C",
        "D"
    )

    graph.add_edge(
        "E",
        "F"
    )

    # ---------------------------------------------------------
    # Print adjacency
    # ---------------------------------------------------------

    print("Adjacency:")
    print(graph._build_adjacency())

    # ---------------------------------------------------------
    # Print longest path
    # ---------------------------------------------------------

    graph.print_longest_root_to_leaf_path()

    # ---------------------------------------------------------
    # Run graph
    # ---------------------------------------------------------

    print("\nExecution")
    print("-" * 40)

    result = graph.run(
        {
            "value": 0
        },
        max_steps=3,
        trace=True
    )

    print("\nFinal state:")
    print(result)
