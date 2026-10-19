import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    # matrices: (p, n, n), vectors: (p, n, 1)
    # Умножаем матрицы на векторы по батчам и суммируем по оси 0
    products = np.matmul(matrices, vectors)
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)
    row_vars = np.var(matrix, axis=1)
    col_vars = np.var(matrix, axis=0)
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        col_means=col_means,
        row_vars=row_vars,
        col_vars=col_vars
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    # Создаем сетку индексов
    r = np.arange(rows)[:, None]
    c = np.arange(columns)[None, :]
    # Чередование по сумме индексов
    mask = (r + c) % 2 == 0
    matrix = np.empty((rows, columns), dtype=type(first))
    matrix[mask] = first
    matrix[~mask] = second
    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    
    center_y, center_x = image_height // 2, image_width // 2
    y1 = max(0, center_y - height // 2)
    y2 = min(image_height, center_y + height // 2 + (height % 2))
    x1 = max(0, center_x - width // 2)
    x2 = min(image_width, center_x + width // 2 + (width % 2))
    
    image[y1:y2, x1:x2] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    
    center_y, center_x = (image_height - 1) / 2.0, (image_width - 1) / 2.0
    y, x = np.ogrid[:image_height, :image_width]
    
    # Уравнение эллипса
    mask = ((x - center_x) ** 2) / (semi_axis_x ** 2 + 1e-12) + ((y - center_y) ** 2) / (semi_axis_y ** 2 + 1e-12) <= 1.0
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    arr = np.asarray(values, dtype=float)
    
    mean_val = np.mean(arr)
    var_val = np.var(arr)
    std_val = np.std(arr)
    
    # Локальные экстремумы (без учета первого и последнего элементов)
    if len(arr) > 2:
        local_max = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1
        local_min = np.where((arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:]))[0] + 1
    else:
        local_max = np.array([], dtype=int)
        local_min = np.array([], dtype=int)
        
    # Скользящее среднее
    weights = np.ones(window) / window
    moving_avg = np.convolve(arr, weights, mode='valid')
    
    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std_deviation=std_val,
        local_maxima=local_max.tolist(),
        local_minima=local_min.tolist(),
        moving_average=moving_avg
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    arr = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = np.max(arr) + 1 if arr.size > 0 else 0
        
    result = np.zeros((arr.size, class_count), dtype=int)
    if arr.size > 0:
        result[np.arange(arr.size), arr] = 1
    return result