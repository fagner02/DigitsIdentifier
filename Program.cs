using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace Digits
{
    class BackpropagationResponse
    {
        public List<List<double>> Biases;
        public List<List<List<double>>> Weights;
        public BackpropagationResponse(List<List<double>> biases, List<List<List<double>>> weights)
        {
            Biases = biases;
            Weights = weights;
        }
    }
    class Network
    {
        public List<int> sizes;
        public List<List<int>> layers;
        public List<List<List<double>>> weights;
        public List<List<double>> biases;
        public List<List<double>> gradients;
        public List<List<int>> tests;
        public List<int> tests_results;
        public List<List<int>> input;
        public double progress = 1;
        public List<double> globalCost = Enumerable.Repeat(1.0, 10).ToList();
        public Random rand = new Random();
        public Network(string? filename = null)
        {
            sizes = new List<int>() { 784, 30, 16, 10 };

            layers = new List<List<int>>() {
                Enumerable.Repeat(0, sizes[0]).ToList(),
                Enumerable.Repeat(0, sizes[1]).ToList(),
                Enumerable.Repeat(0, sizes[2]).ToList()
            };

            weights = new();
            for (int k = 0; k < sizes.Count - 1; k++)
            {
                List<List<double>> result = new();
                for (int i = 0; i < sizes[k + 1]; i++)
                {
                    result.Add(new List<double>());
                    for (int j = 0; j < sizes[k]; j++)
                    {
                        result[i].Add(0.0);
                    }
                }
                weights.Add(result);
            }

            biases = new List<List<double>>();

            for (int i = 0; i < sizes.Count - 1; i++)
            {
                biases.Add(Enumerable.Repeat(0.0, sizes[i + 1]).ToList());
            }

            if (filename == null)
            {
                for (int i = 0; i < weights.Count; i++)
                {
                    for (int j = 0; j < weights[i].Count; j++)
                    {
                        for (int k = 0; k < weights[i][j].Count; k++)
                        {
                            weights[i][j][k] = rand.NextDouble();
                        }

                        biases[i][j] = rand.NextDouble();
                    }
                }
                Save();
                return;
            }

            using var sr = new StreamReader(@"C:\Users\Fagner\source\repos\digits\save.txt");
            for (int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < weights[i].Count; j++)
                {
                    double temp;
                    for (int k = 0; k < weights[i][j].Count; k++)
                    {
                        double.TryParse(sr.ReadLine(), out temp);
                        weights[i][j][k] = temp;
                    }

                    double.TryParse(sr.ReadLine(), out temp);
                    biases[i][j] = temp;
                }
            }
        }

        public void Save()
        {
            StreamWriter sw = new(@"C:\Users\Fagner\source\repos\digits\save.txt");
            string s = "";
            for (int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < weights[i].Count; j++)
                {
                    for (int k = 0; k < weights[i][j].Count; k++)
                    {
                        s += weights[i][j][k].ToString() + '\n';
                    }
                    s += biases[i][j].ToString() + '\n';
                }
            }
            sw.WriteLine(s);
            sw.Close();
            Console.WriteLine(globalCost);
        }

        public List<double> FeedForward(List<double> layer)
        {

            for (int i = 0; i < weights.Count; i++)
            {
                var newLayer = new List<double>();
                for (int j = 0; j < weights[i].Count; j++)
                {
                    double sigmoidOut = 0;
                    for (int k = 0; k < weights[i][j].Count; k++)
                    {
                        sigmoidOut += weights[i][j][k] * layer[k];
                    }
                    newLayer.Add(sigmoidOut + biases[i][j]);
                }
                layer = Sigmoid(newLayer);
            }
            return layer;
        }

        public List<double> Sigmoid(List<double> layer)
        {
            var newLayer = Enumerable.Repeat(0.0, layer.Count).ToList();
            for (int i = 0; i < layer.Count; i++)
            {
                newLayer[i] = 1.0 / (1.0 + Math.Exp(-layer[i]));
            }
            return newLayer;
        }

        public void StochasticGradientDescent(List<List<double>> data, List<List<double>> expectedResults, int epochs, double learnigRate, int batchSize, bool test = false)
        {
            if (test)
            {
                Console.WriteLine("test");
                for (int x = 0; x < data.Count; x++)
                {
                    var res = FeedForward(data[x]);
                    Console.WriteLine(res.IndexOf(res.Max()).ToString() + " " + expectedResults[x].IndexOf(expectedResults[x].Max()));
                }
                return;
            }

            int dataSize = data.Count;

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < (data.Count / batchSize); j++)
                {
                    List<List<double>> batches = new();
                    List<List<double>> batchesRes = new();
                    for (int k = 0; k < batchSize; k ++)
                    {
                        var index = Convert.ToInt32((data.Count - 1) * rand.NextDouble());
                        batches.Add(data[index]);
                        batchesRes.Add(expectedResults[index]);
                    }

                    UpdateNetworkValues(batches, batchesRes, learnigRate);
                    Console.WriteLine("batch");
                }
                Save();
            }
            
        }

        public void UpdateNetworkValues(List<List<double>> batches, List<List<double>> expectedResults, double learningRate)
        {
            //    nabla_b = [np.zeros(b.shape) for b in self.biases]
            //nabla_w = [np.zeros(w.shape) for w in self.weights]

            var newWeights = new List<List<List<double>>>();

            for (int k = 0; k < sizes.Count - 1; k++)
            {
                List<List<double>> result = new();
                for (int i = 0; i < sizes[k + 1]; i++)
                {
                    result.Add(new List<double>());
                    for (int j = 0; j < sizes[k]; j++)
                    {
                        result[i].Add(0.0);
                    }
                }
                newWeights.Add(result);
            }

            var newBiases = new List<List<double>>();
            for (int i = 0; i < sizes.Count - 1; i++)
            {
                newBiases.Add(Enumerable.Repeat(0.0, sizes[i + 1]).Cast<double>().ToList());
            }

            for (int n = 0; n < batches.Count; n++)
            {
                var result = Backpropagation(batches[n], expectedResults[n]);
                for (int i = 0; i < newWeights.Count; i++)
                {
                    for (int j = 0; j < newWeights[i].Count; j++)
                    {
                        for (int k = 0; k < newWeights[i][j].Count; k++)
                        {
                            newWeights[i][j][k] += result.Weights[i][j][k];
                        }
                        newBiases[i][j] += result.Biases[i][j];
                    }
                }
            }

            for (int i = 0; i < newWeights.Count; i++)
            {
                for (int j = 0; j < newWeights[i].Count; j++)
                {
                    for (int k = 0; k < newWeights[i][j].Count; k++)
                    {
                        newWeights[i][j][k] *= (learningRate / batches.Count);
                    }
                    newBiases[i][j] *= (learningRate / batches.Count);
                }
            }

            for (int i = 0; i < newWeights.Count; i++)
            {
                for (int j = 0; j < newWeights[i].Count; j++)
                {
                    for (int k = 0; k < newWeights[i][j].Count; k++)
                    {
                        weights[i][j][k] -= newWeights[i][j][k];
                    }
                    biases[i][j] -= newBiases[i][j];
                }
            }
        }

        public BackpropagationResponse Backpropagation(List<double> input, List<double> expectedResult)
        {
            var activations = new List<List<double>>() { input };

            var zValues = new List<List<double>>();

            for (int i = 0; i < weights.Count; i++)
            {
                var layer = new List<double>();
                for (int j = 0; j < weights[i].Count; j++)
                {
                    double sigmoidOut = 0;
                    for (int k = 0; k < weights[i][j].Count; k++)
                    {
                        sigmoidOut += weights[i][j][k] * activations[i][k];
                    }
                    layer.Add((sigmoidOut/10.0) + biases[i][j]);
                }

                zValues.Add(layer);

                activations.Add(Sigmoid(layer));
            }

            var cost = CostDerivatives(activations.Last(), expectedResult);
            globalCost = CostDerivatives(activations.Last(), expectedResult);
            Console.WriteLine(activations.Last().IndexOf(activations.Last().Max()).ToString() + " " + expectedResult.IndexOf(expectedResult.Max()));

            if (cost.Average(x => Math.Abs(x)) < progress)
            {
                progress = cost.Average(x => Math.Abs(x));
                StreamWriter sw = new StreamWriter(@"C:\Users\Fagner\source\repos\digits\progress.txt", true);
                sw.WriteLine(progress);
                sw.Close();
            }

            for (int i = 0; i < cost.Count; i++)
            {
                cost[i] = cost[i] * SigmoidDerivative(zValues.Last())[i];
            }

            var newBiases = new List<List<double>>();
            for (int i = 0; i < biases.Count; i++)
            {
                newBiases.Add(new List<double>());
            }

            var newWeights = new List<List<List<double>>>();
            for (int i = 0; i < weights.Count; i++)
            {
                newWeights.Add(new List<List<double>>());
            }

            newBiases[newBiases.Count - 1] = cost;

            for (int j = 0; j < cost.Count; j++)
            {
                var layerWeights = new List<double>(activations[activations.Count - 2]);
                for (int i = 0; i < layerWeights.Count; i++)
                {
                    layerWeights[i] *= cost[j];
                }
                newWeights[newWeights.Count - 1].Add(layerWeights);
            }

            for (int i = 2; i <= weights.Count; i++)
            {
                var z = zValues[zValues.Count - i];
                var sigmoidDerivatives = SigmoidDerivative(z);

                var layer = new List<double>();
                var transposed = Transpose(weights[weights.Count - i + 1]);
                for (int j = 0; j < transposed.Count; j++)
                {
                    double sigmoidOut = 0;
                    for (int k = 0; k < transposed[j].Count; k++)
                    {
                        sigmoidOut += transposed[j][k] * cost[k];
                    }
                    layer.Add(sigmoidOut * sigmoidDerivatives[j]);
                }
                cost = layer;

                newBiases[newBiases.Count - i] = cost;

                for (int k = 0; k < cost.Count; k++)
                {
                    var layerWeights = new List<double>(activations[activations.Count - i - 1]);
                    for (int n = 0; n < layerWeights.Count; n++)
                    {
                        layerWeights[n] *= cost[k];
                    }
                    newWeights[newWeights.Count - i].Add(layerWeights);
                }
            }

            return new BackpropagationResponse(newBiases, newWeights);
        }

        public List<List<double>> Transpose(List<List<double>> matrix)
        {
            int w = matrix.Count;
            int h = matrix[matrix.Count - 1].Count;

            List<List<double>> result = new();
            for (int i = 0; i < h; i++)
            {
                result.Add(new List<double>());
                for (int j = 0; j < w; j++)
                {
                    result[i].Add(0.0);
                }
            }

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j][i] = matrix[i][j];
                }
            }

            return result;
        }

        public List<double> SigmoidDerivative(List<double> sigmoids)
        {
            // (e^-z)/((1+e^-z)^2)
            var derivatives = new List<double>();

            var sigmoidFunctionResult = Sigmoid(sigmoids);

            for (int i = 0; i < sigmoidFunctionResult.Count; i++)
            {
                derivatives.Add(sigmoidFunctionResult[i] * (1.0 - sigmoidFunctionResult[i]));
            }
            return derivatives;
        }

        public List<double> CostDerivatives(List<double> actualResult, List<double> expectedResult)
        {
            var result = new List<double>();
            for (int i = 0; i < actualResult.Count; i++)
            {
                result.Add(actualResult[i] - expectedResult[i]);
            }

            return result;
        }

        public void Evaluate(List<double> tests)
        {
            for (int i = 0; i < tests.Count; i++)
            {
                FeedForward(tests);
                for (int j = 0; j < layers[i + 1].Count; j++)
                {
                    //layers[i + 1][j] = Sigmoid(layers[i][j] * weights[i][j] + biases[i][j]);
                }
            }
        }
    }
    class response
    {
        public List<List<double>> Data;
        public List<List<double>> Result;
        public response(List<List<double>> data, List<List<double>> result)
        {
            Data = data;
            Result = result;
        }

    }
    class Loader
    {
        private readonly List<double> chars = new() { 32, 46, 45, 42, 43, 35, 64, 72, 69 };
        public response Load(int take)
        {
            Console.WriteLine("\nBegin\n");

            FileStream ifsLabels =
             new FileStream(@"C:\Users\Fagner\source\repos\digits\t10k-labels.idx1-ubyte",
             FileMode.Open); // test labels

            FileStream ifsImages =
             new FileStream(@"C:\Users\Fagner\source\repos\digits\t10k-images.idx3-ubyte",
             FileMode.Open); // test images

            BinaryReader brLabels = new BinaryReader(ifsLabels);

            BinaryReader brImages = new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            var results = new List<List<double>>();
            var result = new List<List<double>>();
            for (int di = 0; di < take; di++)
            {
                var pixels = new List<double>();
                for (int i = 0; i < 784; i++)
                {
                    pixels.Add(brImages.ReadByte() / 255.0);
                }
                result.Add(pixels);

                results.Add(Enumerable.Repeat(0.0, 10).ToList());
                results[results.Count - 1][brLabels.ReadByte()] = 1;
            }

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            return new response(result, results);
        }
        public void Print(BinaryReader brImages, BinaryReader brLabels)
        {
            // each test image
            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];

            for (int di = 0; di < 2; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }

                byte lbl = brLabels.ReadByte();


                string s = "";
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        s += (char)chars[(int)((pixels[i][j] / 255.0) * (chars.Count - 1))];
                    }
                    s += "\n";
                }

                Console.WriteLine(s);
            } // each image

            Console.WriteLine("\nEnd\n");
            Console.ReadLine();
        }
    }

    class Program
    {
        static void Main()
        {
            Loader loader = new Loader();
            var result = loader.Load(10000);

            var net = new Network();
            for (int i = 0; i < 1; i++)
            {
                net.StochasticGradientDescent(result.Data, result.Result, 30, 3, 10);
            }
        }
    }
}