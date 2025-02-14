export interface ILayer {
  forward(input: number[][]): number[][]
  backward(gradOutput: number[][], learningRate: number): number[][]
}
