export function tanh(x: number): number {
  return Math.tanh(x)
}

export function relu(x: number): number {
  return Math.max(0, x)
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}
