export function multiplyMatrixVector(matrix: number[][], vector: number[]): number[] {
  return matrix.map(row =>
    row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
  );
}

export function addVectors(vec1: number[], vec2: number[]): number[] {
  return vec1.map((val, idx) => val + vec2[idx]);
}

export function initializeMatrix(rows: number, cols: number, initFunc: () => number = () => Math.random() * 0.1): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, initFunc)
  );
}

export function initializeVector(size: number, initFunc: () => number = () => Math.random() * 0.1): number[] {
  return Array.from({ length: size }, initFunc);
}
