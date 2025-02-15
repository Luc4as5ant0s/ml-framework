import { RNNModel } from "./models/rnn"

function generateData(numSamples: number, sequenceLength: number) {
  const data: { sequence: number[][]; target: number[][] }[] = []
  for (let i = 0; i < numSamples; i++) {
    const start = Math.random()
    const sequence: number[][] = []
    const target: number[][] = []
    for (let t = 0; t < sequenceLength; t++) {
      sequence.push([start + t * 0.1])
      target.push([start + (t + 1) * 0.1])
    }
    data.push({ sequence, target })
  }
  return data
}

const numSamples = 100
const sequenceLength = 5
const trainingData = generateData(numSamples, sequenceLength)

const inputSize = 1
const hiddenSize = 20
const outputSize = 1

const rnn = new RNNModel(inputSize, hiddenSize, outputSize)

const learningRate = 0.0006
const epochs = 200000

for (let epoch = 0; epoch < epochs; epoch++) {
  let epochLoss = 0
  for (const sample of trainingData) {
    const loss = rnn.train(sample.sequence, sample.target, learningRate)
    epochLoss += loss
  }
  epochLoss /= numSamples
  if (epoch % 20 === 0) {
    console.log(`Epoch ${epoch}: Loss = ${epochLoss.toFixed(4)}`)
  }
}

const testSequence = [[0.5], [0.6], [0.7], [0.8], [0.9]]
const predicted = rnn.forward(testSequence)
console.log("Test sequence:", testSequence)
console.log("Predicted next values:", predicted)
