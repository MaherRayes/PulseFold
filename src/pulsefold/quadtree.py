import heapq
from pathlib import Path

import numpy as np

from .config import FRAME_CHUNK_SIZE
from .frameloader import FrameStack

GridCell = tuple[int, int, int, int]

def _nodes_children(nodes: list[GridCell]) -> list[GridCell]:

    all_children = []
    for y0, y1, x0, x1 in nodes:
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue

        all_children.append((y0, y0 + (y1 - y0) // 2, x0, x0 + (x1 - x0) // 2))
        all_children.append((y0 + (y1 - y0) // 2, y1, x0, x0 + (x1 - x0) // 2))
        all_children.append((y0, y0 + (y1 - y0) // 2, x0 + (x1 - x0) // 2, x1))
        all_children.append((y0 + (y1 - y0) // 2, y1, x0 + (x1 - x0) // 2, x1))

    return all_children

def _chunked_array_sum(frames: np.ndarray, frame_chunk: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    array_sum = np.zeros(frames.shape[1:], dtype=np.float64)
    array_sum_squared = np.zeros(frames.shape[1:], dtype=np.float64)
    for i in range(0, frame_count, frame_chunk):
        j = min(i + frame_chunk, frame_count)
        chunk = frames[i:j].astype(np.float64)
        array_sum += np.sum(chunk, axis=0)
        array_sum_squared += np.sum(chunk * chunk, axis=0)
    return array_sum, array_sum_squared

def _2d_prefix_sum_nd_array(frames_array: np.ndarray) -> np.ndarray:
    prefix_array = np.cumsum(np.cumsum(frames_array, axis=-2, dtype=np.float64), axis=-1)
    return np.pad(prefix_array, [(0, 0)] * (prefix_array.ndim - 2) + [(1, 0), (1, 0)])

def _get_sums_from_prefix(prefix_array: np.ndarray, nodes: list[GridCell]) -> np.ndarray:
    y0 = np.fromiter((node[0] for node in nodes), dtype=np.intp, count=len(nodes))
    y1 = np.fromiter((node[1] for node in nodes), dtype=np.intp, count=len(nodes))
    x0 = np.fromiter((node[2] for node in nodes), dtype=np.intp, count=len(nodes))
    x1 = np.fromiter((node[3] for node in nodes), dtype=np.intp, count=len(nodes))
    return prefix_array[..., y1, x1] - prefix_array[..., y0, x1] - prefix_array[..., y1, x0] + prefix_array[..., y0, x0]

def _chunked_nodes_sum_squares(frames: np.ndarray, counts: np.ndarray, nodes: list[GridCell], frame_chunk: int, frame_count: int) -> np.ndarray:
    result = np.zeros(len(nodes), dtype=np.float64)
    for i in range(0, frame_count, frame_chunk):
        j = min(i + frame_chunk, frame_count)
        prefix_sum_squares = _2d_prefix_sum_nd_array(frames[i:j].astype(np.float64) * counts)
        sums = _get_sums_from_prefix(prefix_sum_squares, nodes)
        result += np.einsum("ij,ij->j", sums, sums)
    return result

def _compute_nodes_sse(frames: FrameStack, counts: np.ndarray, root: GridCell) -> dict[GridCell, tuple[float, float]]:
    nodes = [root]
    all_sse:dict[GridCell, tuple[float, float]] = {}
    frame_count = frames.frame_count
    frame_chunk = max(1, min(FRAME_CHUNK_SIZE, frame_count))

    frames_sum, frames_sum_squared = _chunked_array_sum(frames.frames, frame_chunk, frame_count)

    prefix_area = _2d_prefix_sum_nd_array(counts)
    prefix_energy = _2d_prefix_sum_nd_array(frames_sum_squared * counts)
    prefix_static_energy = _2d_prefix_sum_nd_array(np.square(frames_sum) * counts)
    prefix_static_weighted_sum = _2d_prefix_sum_nd_array(frames_sum * counts)

    while nodes:
        next_nodes = []

        area = _get_sums_from_prefix(prefix_area, nodes)
        sum_squares = _chunked_nodes_sum_squares(frames.frames, counts, nodes, frame_chunk, frame_count)
        energy = _get_sums_from_prefix(prefix_energy, nodes)
        static_energy = _get_sums_from_prefix(prefix_static_energy, nodes)
        static_weighted_sum = _get_sums_from_prefix(prefix_static_weighted_sum, nodes)

        node_total_sse = np.maximum(np.zeros([len(nodes)]), energy - sum_squares / area)
        node_static_sse = np.maximum(np.zeros([len(nodes)]), (static_energy / frame_count) - (np.square(static_weighted_sum) / (frame_count * area)))
        node_temporal_sse = np.maximum(np.zeros([len(nodes)]), node_total_sse - node_static_sse)
        all_sse.update(zip(nodes, list(zip(node_static_sse, node_temporal_sse))))

        next_nodes.extend(_nodes_children(nodes))
        nodes = next_nodes

    return all_sse

def _node_area(node: GridCell) -> int:
    return (node[1] - node[0]) * (node[3] - node[2])

def _build_unguided_quadtree(nodes: list[GridCell], leaf_budget: int) -> tuple[list[GridCell], int]:
    leaves = set(nodes)
    node_queue: list[tuple[int, int, GridCell]] = []
    node_counter = 0

    for node in nodes:
        heapq.heappush(node_queue, (-_node_area(node), node_counter, node))
        node_counter += 1
    
    while node_queue and len(leaves) < leaf_budget:
        _, _, node = heapq.heappop(node_queue)
        if node not in leaves:
            continue

        children = _nodes_children([node])
        if not children or len(leaves) + len(children) - 1 > leaf_budget:
            continue

        leaves.remove(node)
        for child in children:
            leaves.add(child)
            heapq.heappush(node_queue, (-_node_area(child), node_counter, child))
            node_counter += 1

    return leaves, node_counter

def _guided_enqueue(node: GridCell, errors: dict[GridCell, (float, float)], queue: list[tuple[float, int, GridCell]], temporal_weight: float, counter: int) -> None:
    children = _nodes_children([node])
    if not children:
        return counter

    #Calculate error difference if we split the node
    parent_error = errors[node]
    static_diff = max(0.0, parent_error[0] - sum(errors[child][0] for child in children))
    temporal_diff = max(0.0, parent_error[1] - sum(errors[child][1] for child in children))
    error_diff = static_diff + temporal_weight**2 * temporal_diff

    if error_diff > 0:
        heapq.heappush(queue, (-error_diff, counter, node))
        counter += 1


def _build_guided_quadtree(nodes: list[GridCell], errors: dict[GridCell, (float, float)], leaf_budget: int, temporal_weight: float, counter: int) -> list[GridCell]:
    leaves = set(nodes)
    node_queue: list[tuple[float, int, GridCell]] = []
    node_counter = counter

    for leaf in leaves:
        _guided_enqueue(leaf, errors, node_queue, temporal_weight, node_counter)

    while node_queue and len(leaves) < leaf_budget:
        _, _, node = heapq.heappop(node_queue)
        if node not in leaves:
            continue

        children = _nodes_children([node])
        if not children or len(leaves) + len(children) > leaf_budget:
            continue

        leaves.remove(node)
        for child in children:
            leaves.add(child)
            _guided_enqueue(child, errors, node_queue, temporal_weight, node_counter)

    return leaves


def build_quadtree(frames: FrameStack, base_cell_factor: int, leaf_count: int, unguided_node_budget: float, temporal_weight: float, output_path: Path) -> tuple[np.memmap, np.ndarray, np.ndarray]:

    if base_cell_factor < 1:
        raise ValueError("Base cell factor has to be 1 or higher.")
    if leaf_count < 1:
        raise ValueError("Leaf count has to be 1 or higher.")
    if unguided_node_budget < 0.0 or unguided_node_budget > 1.0:
        raise ValueError("Unguided node percentage has to be between 0.0 and 1.0.")
    if temporal_weight < 0.0:
        raise ValueError("Temporal weight has to be 0.0 or higher.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    counts = frames.counts
    root:GridCell = (0, frames.reduced_height, 0, frames.reduced_width)

    errors = _compute_nodes_sse(frames, counts, root)
    print("Finished calculating nodes sse")
    nodes, counter = _build_unguided_quadtree([root], int(leaf_count * unguided_node_budget))
    leaves = _build_guided_quadtree(nodes, errors, leaf_count, temporal_weight, counter)
    leaves = sorted(leaves)

    gridcells = np.empty((len(leaves), 4), dtype=np.uint32)
    for index, (y0, y1, x0, x1) in enumerate(leaves):
        gridcells[index] = (y0 * base_cell_factor, min(y1 * base_cell_factor, frames.original_height),
                            x0 * base_cell_factor, min(x1 * base_cell_factor, frames.original_width))

    output = np.memmap(output_path, mode="w+", dtype=np.float32, shape=(frames.frame_count, len(leaves)))

    prefix_areas = _2d_prefix_sum_nd_array(counts)
    areas = _get_sums_from_prefix(prefix_areas, leaves)
    frame_chunk = max(1, min(FRAME_CHUNK_SIZE, frames.frame_count))
    for i in range(0, frames.frame_count, frame_chunk):
        j = min(i + frame_chunk, frames.frame_count)
        prefix = _2d_prefix_sum_nd_array(frames.frames[i:j] * counts)
        output[i:j] += _get_sums_from_prefix(prefix, leaves) / areas

    output.flush()

    return output, gridcells, counts