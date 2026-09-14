import heapq
from pathlib import Path

import numpy as np

from .frameloader import FrameStack

GridCell = tuple[int, int, int, int]

def _spatial_average(frames: FrameStack, factor: int, counts: np.ndarray, dtype=np.float64) -> FrameStack:
    array = np.asarray(frames.frames).astype(dtype)
    y_indicies = np.arange(0, frames.height, factor)
    x_indicies = np.arange(0, frames.width, factor)

    frames.frames = np.add.reduceat(np.add.reduceat(array, y_indicies, axis=1), x_indicies, axis=2) / counts
    _, frames.height, frames.width = frames.frames.shape

    return frames

def _factored_pixel_counts(height: int, width: int, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("spatial factor must be at least 1")

    row_counts = np.full((height + factor - 1) // factor, factor, dtype=np.int32)
    column_counts = np.full((width + factor - 1) // factor, factor, dtype=np.int32)

    row_counts[-1] = height % factor if height % factor > 0 else factor
    column_counts[-1] = width % factor if width % factor > 0 else factor

    return np.outer(row_counts, column_counts)

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

def _2d_prefix_sum_nd_array(frames_array: np.ndarray) -> np.ndarray:
    prefix_array = np.cumsum(np.cumsum(frames_array, axis=-2), axis=-1)
    return np.pad(prefix_array, [(0, 0)] * (prefix_array.ndim - 2) + [(1, 0), (1, 0)])

def _get_sums_from_prefix(prefix_array: np.ndarray, nodes: GridCell) -> float:
    y0 = np.fromiter((node[0] for node in nodes), dtype=np.intp, count=len(nodes))
    y1 = np.fromiter((node[1] for node in nodes), dtype=np.intp, count=len(nodes))
    x0 = np.fromiter((node[2] for node in nodes), dtype=np.intp, count=len(nodes))
    x1 = np.fromiter((node[3] for node in nodes), dtype=np.intp, count=len(nodes))
    return prefix_array[..., y1, x1] - prefix_array[..., y0, x1] - prefix_array[..., y1, x0] + prefix_array[..., y0, x0]

def _compute_nodes_sse_fast(frames: FrameStack, counts: np.ndarray, root: GridCell) -> dict[GridCell, (float, float)]:
    nodes = [root]
    all_sse:dict[GridCell, (float, float)] = {}

    prefix_area = _2d_prefix_sum_nd_array(counts)
    prefix_weighted_sum = _2d_prefix_sum_nd_array(frames.frames * counts)
    prefix_energy = _2d_prefix_sum_nd_array(np.square(frames.frames) * counts)
    prefix_static_energy = _2d_prefix_sum_nd_array(np.square(np.sum(frames.frames, axis=0)) * counts)
    prefix_static_weighted_sum = _2d_prefix_sum_nd_array(np.sum(frames.frames, axis=0) * counts)

    while nodes:
        next_nodes = []

        area = _get_sums_from_prefix(prefix_area, nodes)
        weighted_sum = _get_sums_from_prefix(prefix_weighted_sum, nodes)
        energy = _get_sums_from_prefix(prefix_energy, nodes)
        static_energy = _get_sums_from_prefix(prefix_static_energy, nodes)
        static_weighted_sum = _get_sums_from_prefix(prefix_static_weighted_sum, nodes)

        node_total_sse = np.maximum(np.full([len(nodes)], 0.0), np.sum(energy) - np.sum(np.square(weighted_sum)) / area)
        node_static_sse = np.maximum(np.full([len(nodes)], 0.0), (static_energy / frames.count) - (np.square(static_weighted_sum) / (frames.count * area)))
        node_temporal_sse = np.maximum(np.full([len(nodes)], 0.0), node_total_sse - node_static_sse)
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


def build_quadtree(frames: FrameStack, base_cell_factor: int, leaf_count: int, unguided_node_budget: float, temporal_weight: float, output_path: Path) -> tuple[np.memmap, np.ndarray]:

    counts = _factored_pixel_counts(frames.height, frames.width, base_cell_factor)
    base_frames = _spatial_average(frames, base_cell_factor, counts)

    root:GridCell = (0, base_frames.height, 0, base_frames.width)

    errors = _compute_nodes_sse_fast(base_frames, counts, root)
    nodes, counter = _build_unguided_quadtree([root], int(leaf_count * unguided_node_budget))
    leaves = _build_guided_quadtree(nodes, errors, leaf_count, temporal_weight, counter)
    leaves = sorted(leaves)

    gridcells = np.empty((len(leaves), 4), dtype=np.uint32)
    for index, (y0, y1, x0, x1) in enumerate(leaves):
        gridcells[index] = (y0 * base_cell_factor, min(y1 * base_cell_factor, frames.height),
                            x0 * base_cell_factor, min(x1 * base_cell_factor, frames.width))

    output = np.memmap(output_path, mode="w+", dtype=np.float32, shape=(frames.count, len(leaves)))

    prefix = _2d_prefix_sum_nd_array(base_frames.frames * counts)
    areas = _2d_prefix_sum_nd_array(counts)
    output[:] = _get_sums_from_prefix(prefix, leaves) / _get_sums_from_prefix(areas, leaves)

    output.flush()

    return output, gridcells