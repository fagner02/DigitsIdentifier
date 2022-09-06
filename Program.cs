using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Digits
{
    class BackpropagationResponse
    {
        public List<List<double>> Biases;
        public List<List<List<double>>> Weights;
        public List<double> Dz3;
        public BackpropagationResponse(List<List<double>> biases, List<List<List<double>>> weights, List<double> dz3)
        {
            Biases = biases;
            Weights = weights;
            Dz3 = dz3;
        }
    }
    class ConvBackpropagationResponse
    {
        public List<List<double>> Biases;
        public List<List<List<List<double>>>> Filters;
        public ConvBackpropagationResponse(List<List<double>> biases, List<List<List<List<double>>>> filters)
        {
            Biases = biases;
            Filters = filters;
        }
    }
    class Network
    {
        public List<int> sizes;
        public List<int> convolutionalSizes;
        public List<List<int>> layers;
        public List<List<List<double>>> weights;
        public List<List<List<List<double>>>> filters;
        public List<List<double>> biases;
        public List<List<double>> convolutionBiases;
        public int lines = 28;
        public int columns = 28;
        public int poolSize = 2;
        public int stride = 2;
        public int filterSize = 3;
        public double progress = 1;
        public List<double> globalCost = Enumerable.Repeat(1.0, 10).ToList();
        public Random rand = new Random();
        public Network(string? filename = null)
        {

            convolutionalSizes = new List<int>() { 9, 5 };

            var inp = new List<int>() { lines };

            for (int i = 0; i < convolutionalSizes.Count; i++)
            {
                var l = new List<int>();
                for (int j = 0; j < inp.Count; j++)
                {
                    for (int k = 0; k < convolutionalSizes[i]; k++)
                    {
                        l.Add((int)Math.Ceiling((double)(inp[j] - filterSize + 1) / stride));
                    }
                }
                inp = l;
            }

            int size = 0;
            for (int i = 0; i < inp.Count; i++)
            {
                size += inp[i] * inp[i];
            }

            sizes = new List<int>() { size, 10 };

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

                filters = new List<List<List<List<double>>>>();
                convolutionBiases = new List<List<double>>();
                for (int i = 0; i < convolutionalSizes.Count; i++)
                {
                    filters.Add(new List<List<List<double>>>());
                    convolutionBiases.Add(new List<double>());
                    for (int n = 0; n < convolutionalSizes[i]; n++)
                    {
                        filters[i].Add(new List<List<double>>());
                        for (int j = 0; j < filterSize; j++)
                        {
                            filters[i][n].Add(new List<double>());
                            for (int k = 0; k < filterSize; k++)
                            {
                                filters[i][n][j].Add(rand.NextDouble());
                            }
                        }
                        convolutionBiases[i].Add(rand.NextDouble());
                    }
                }
                Save();
                return;
            }

            using (var sr = File.OpenRead(filename))
            {
                var br = new BinaryReader(sr);
                for (int i = 0; i < weights.Count; i++)
                {
                    for (int j = 0; j < weights[i].Count; j++)
                    {
                        for (int k = 0; k < weights[i][j].Count; k++)
                        {
                            weights[i][j][k] = br.ReadDouble();
                        }
                        biases[i][j] = br.ReadDouble();
                    }
                }
            }
            using (var sr = File.OpenRead("conv-save.txt"))
            {
                var br = new BinaryReader(sr);
                filters = new List<List<List<List<double>>>>();
                convolutionBiases = new List<List<double>>();
                for (int i = 0; i < convolutionalSizes.Count; i++)
                {
                    filters.Add(new List<List<List<double>>>());
                    convolutionBiases.Add(new List<double>());
                    for (int n = 0; n < convolutionalSizes[i]; n++)
                    {
                        filters[i].Add(new List<List<double>>());
                        for (int j = 0; j < filterSize; j++)
                        {
                            filters[i][n].Add(new List<double>());
                            for (int k = 0; k < filterSize; k++)
                            {
                                filters[i][n][j].Add(br.ReadDouble());
                            }
                        }
                        convolutionBiases[i].Add(br.ReadDouble());
                    }
                }
            }
        }

        public void Print()
        {
            Console.WriteLine("weights");
            weights
                .ForEach(i =>
                    i.ForEach(j =>
                        j.ForEach(k => Console.Write(k.ToString() + ", "))));

            Console.WriteLine("biases");
            biases
                .ForEach(i =>
                    i.ForEach(j => Console.Write(j.ToString() + ", ")));
        }

        public void Save()
        {
            var sw = File.Open(@"save.txt", FileMode.Create);
            BinaryWriter bw = new BinaryWriter(sw);
            string s = "";
            for (int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < weights[i].Count; j++)
                {
                    for (int k = 0; k < weights[i][j].Count; k++)
                    {
                        bw.Write(weights[i][j][k]);
                    }
                    bw.Write(biases[i][j]);
                }
            }
            sw.Close();

            sw = new("conv-save.txt", FileMode.Create);
            bw = new(sw);

            for (int i = 0; i < filters.Count; i++)
            {
                for (int n = 0; n < filters[i].Count; n++)
                {
                    for (int j = 0; j < filters[i][n].Count; j++)
                    {
                        for (int k = 0; k < filters[i][n][j].Count; k++)
                        {
                            bw.Write(filters[i][n][j][k]);
                        }
                    }
                    bw.Write(convolutionBiases[i][n]);
                }
            }
            sw.Close();
        }

        public List<double> FeedForward(List<double> layer)
        {
            double max = layer.Select(x => Math.Abs(x)).Max();
            for (int i = 0; i < layer.Count; i++)
            {
                layer[i] /= max;
            }

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
                    newLayer.Add((sigmoidOut / 10.0) + biases[i][j]);
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
                var hit = 0.0;
                Console.WriteLine("test");
                for (int x = 0; x < data.Count; x++)
                {
                    var res = FeedForward(ConvolutionalFeedFoward(data[x]));
                    Console.WriteLine(res.IndexOf(res.Max()).ToString() + " " + expectedResults[x].IndexOf(expectedResults[x].Max()));
                    hit += res.IndexOf(res.Max()) == expectedResults[x].IndexOf(expectedResults[x].Max()) ? 1 : 0;
                }
                Console.WriteLine(hit / data.Count);
                return;
            }

            int dataSize = data.Count;

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < (data.Count / batchSize); j++)
                {
                    List<List<double>> batches = new();
                    List<List<double>> batchesRes = new();
                    for (int k = 0; k < batchSize; k++)
                    {
                        var index = Convert.ToInt32((data.Count - 1) * rand.NextDouble());
                        batches.Add(data[index]);
                        batchesRes.Add(expectedResults[index]);
                    }
                    Save();
                    UpdateNetworkValues(batches, batchesRes, learnigRate);
                    Console.WriteLine("batch");
                }
                Save();
            }

        }

        public void UpdateNetworkValues(List<List<double>> batches, List<List<double>> expectedResults, double learningRate)
        {
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


            ConvBackpropagationResponse convolutionValues = null;
            for (int n = 0; n < batches.Count; n++)
            {
                var result = Backpropagation(ConvolutionalFeedFoward(batches[n]), expectedResults[n]);
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
                if (convolutionValues == null)
                {
                    convolutionValues = ConvolutionalBackpropagation(batches[n], result.Dz3);
                }
                else
                {
                    convolutionValues = SumBackpropagationValues(convolutionValues, ConvolutionalBackpropagation(batches[n], result.Dz3));
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
            var zero = true;
            for (int i = 0; i < convolutionValues.Filters.Count; i++)
            {
                for (int j = 0; j < convolutionValues.Filters[i].Count; j++)
                {
                    for (int k = 0; k < convolutionValues.Filters[i][j].Count; k++)
                    {
                        for (int n = 0; n < convolutionValues.Filters[i][j][k].Count; n++)
                        {
                            if ((learningRate / (batches.Count)) * convolutionValues.Filters[i][j][k][n] != 0)
                            {
                                zero = false;
                            }
                            filters[i][j][k][n] -= (learningRate / (batches.Count)) * convolutionValues.Filters[i][j][k][n];
                        }
                    }
                    convolutionBiases[i][j] -= (learningRate / (100*batches.Count)) * convolutionValues.Biases[i][j];
                }
            }
            Console.WriteLine(zero);
        }

        public ConvBackpropagationResponse SumBackpropagationValues(ConvBackpropagationResponse a, ConvBackpropagationResponse b)
        {
            var result = new ConvBackpropagationResponse(a.Biases, a.Filters);
            for (int i = 0; i < a.Filters.Count; i++)
            {
                for (int j = 0; j < a.Filters[i].Count; j++)
                {
                    for (int k = 0; k < a.Filters[i][j].Count; k++)
                    {
                        for (int n = 0; n < a.Filters[i][j][k].Count; n++)
                        {
                            result.Filters[i][j][k][n] += b.Filters[i][j][k][n];
                        }
                    }
                    result.Biases[i][j] += b.Biases[i][j];
                }
            }
            return result;
        }


        public BackpropagationResponse Backpropagation(List<double> input, List<double> expectedResult)
        {
            double max = input.Select(x => Math.Abs(x)).Max();
            for (int i = 0; i < input.Count; i++)
            {
                input[i] /= max;
            }

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
                    layer.Add((sigmoidOut / 10.0) + biases[i][j]);
                }

                zValues.Add(layer);

                activations.Add(Sigmoid(layer));
            }

            var cost = CostDerivatives(activations.Last(), expectedResult);
            globalCost = CostDerivatives(activations.Last(), expectedResult);
            Console.WriteLine(activations.Last().IndexOf(activations.Last().Max()).ToString() + " " + expectedResult.IndexOf(expectedResult.Max()));
            var dz3 = cost.ConvertAll(x => x);
            if (cost.Average(x => Math.Abs(x)) < progress)
            {
                progress = cost.Average(x => Math.Abs(x));
                StreamWriter sw = new StreamWriter(@"progress.txt", true);
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

            newBiases[^1] = cost;

            for (int j = 0; j < cost.Count; j++)
            {
                var layerWeights = new List<double>(activations[^2]);
                for (int i = 0; i < layerWeights.Count; i++)
                {
                    layerWeights[i] *= cost[j];
                }
                newWeights[^1].Add(layerWeights);
            }

            for (int i = 2; i <= weights.Count; i++)
            {
                var z = zValues[^i];
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

                newBiases[^i] = cost;

                for (int k = 0; k < cost.Count; k++)
                {
                    var layerWeights = new List<double>(activations[activations.Count - i - 1]);
                    for (int n = 0; n < layerWeights.Count; n++)
                    {
                        layerWeights[n] *= cost[k];
                    }
                    newWeights[^i].Add(layerWeights);
                }
            }

            return new BackpropagationResponse(newBiases, newWeights, dz3);
        }

        public List<List<double>> Transpose(List<List<double>> matrix)
        {
            int w = matrix.Count;
            int h = matrix[^1].Count;

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

        public ConvBackpropagationResponse ConvolutionalBackpropagation(List<double> flattenInput, List<double> dz3)
        {
            var input = new List<List<double>>() { };
            for (int i = 0; i < lines; i++)
            {
                input.Add(new List<double>());
                for (int j = 0; j < columns; j++)
                {
                    input[i].Add(flattenInput[i * lines + j]);
                }
            }
            var ps = new List<List<List<List<double>>>>() { new List<List<List<double>>>() { input } };
            var cs = new List<List<List<List<double>>>>();
            var zs = new List<List<List<List<double>>>>();
            for (int i = 0; i < filters.Count; i++)
            {
                zs.Add(new List<List<List<double>>>());
                ps.Add(new List<List<List<double>>>());
                cs.Add(new List<List<List<double>>>());
                for (int j = 0; j < ps[i].Count; j++)
                {
                    for (int k = 0; k < filters[i].Count; k++)
                    {
                        zs.Last().Add(Convolution(ps[i][j], filters[i][k], convolutionBiases[i][k]));
                        cs.Last().Add(ReLu(zs.Last().Last()));
                        ps.Last().Add(Pooling(cs.Last().Last()));
                    }
                }
            }

            var transposed = Transpose(weights.First());

            var df = MatrixMultiplication(new() { transposed }, new() { dz3.ConvertAll(x => new List<double>() { x }) })[0];

            var dp = new List<List<List<double>>>();

            for (int i = 0; i < ps[^1].Count; i++)
            {
                dp.Add(new List<List<double>>());
                for (int j = 0; j < ps[^1][i].Count; j++)
                {
                    dp[i].Add(new List<double>());
                    for (int k = 0; k < ps[^1][i][j].Count; k++)
                    {
                        dp[i][j].Add(df[j * ps[^1][i].Count + k][0]);
                    }
                }
                df = df.Skip(ps[^1][i].Count * ps[^1][i].Count).ToList();
            }

            var dc = DcCalc(cs.Last(), dp);

            var zRect = RectZ(zs.Last());

            var dz = Multiplication(dc, zRect);

            var newFilters = new List<List<List<List<double>>>>(filters);
            newFilters = newFilters.ConvertAll(x => x.ConvertAll(y => y.ConvertAll(k => k.ConvertAll(n => n))));
            newFilters[^1] = BackConvolution(dz, ps[^2], true);

            var newBiases = new List<List<double>>(convolutionBiases);
            newBiases = newBiases.ConvertAll(x => x.ConvertAll(y => y));
            for (int i = 0; i < convolutionBiases[^1].Count; i++)
            {
                var sumLayer = dz.Skip(i * (dz.Count / newBiases[^1].Count)).Take(dz.Count / newBiases[^1].Count).ToList();
                var sum = sumLayer.Select(x => x.Select(y => y.Sum()).Sum()).Sum();
                newBiases[^1][i] = sum / ((dz.Count / newBiases[^1].Count) * dz[0].Count * dz[0][0].Count);
            }

            for (int i = filters.Count - 2; i > -1; i--)
            {
                var paddedZ = AddPadding(zs[i + 1]);

                var rotatedFilters = Rotate(filters[i + 1]);

                dp = BackConvolution(paddedZ, rotatedFilters);

                dc = DcCalc(cs[i], dp);

                zRect = RectZ(zs[i]);

                dz = Multiplication(dc, zRect);

                newFilters[i] = BackConvolution(dz, ps[i], true);

                for (int m = 0; m < convolutionBiases[i].Count; m++)
                {

                    var sumLayer = dz.Skip(m * (dz.Count / newBiases[i].Count)).Take(dz.Count / newBiases[i].Count).ToList();
                    var sum = sumLayer.Select(x => x.Select(y => y.Sum()).Sum()).Sum();
                    newBiases[i][m] = sum / ((dz.Count / newBiases[i].Count) * dz[0].Count * dz[0][0].Count);
                }
            }
            //Console.WriteLine("Propagation done");
            return new ConvBackpropagationResponse(newBiases, newFilters);
        }

        public List<List<List<double>>> Multiplication(List<List<List<double>>> dc, List<List<List<double>>> zRect)
        {
            var result = new List<List<List<double>>>();
            for (int i = 0; i < dc.Count; i++)
            {
                result.Add(new List<List<double>>());
                for (int j = 0; j < dc[i].Count; j++)
                {
                    result[i].Add(new List<double>());
                    for (int k = 0; k < dc[i][j].Count; k++)
                    {
                        result[i][j].Add(dc[i][j][k] * zRect[i][j][k]);
                    }
                }
            }

            return result;
        }

        public List<List<List<double>>> BackConvolution(List<List<List<double>>> paddedZ, List<List<List<double>>> rotatedFilters, bool reverse = false)
        {
            var dp = new List<List<List<double>>>();
            for (int j = 0; j < paddedZ.Count / rotatedFilters.Count; j++)
            {
                var layer = new List<List<List<double>>>(paddedZ.Skip(j * rotatedFilters.Count).Take(rotatedFilters.Count).ToList());
                layer = layer.ConvertAll(x => x.ConvertAll(y => y.ConvertAll(k => k)));

                for (int k = 0; k < layer.Count; k++)
                {
                    if (reverse)
                    {
                        layer[k] = Convolution(rotatedFilters[k], layer[k]);
                        continue;
                    }
                    layer[k] = Convolution(layer[k], rotatedFilters[k]);
                }

                var res = layer.Skip(1).Aggregate(layer.First(), (x, y) =>
                {
                    for (int n = 0; n < x.Count; n++)
                    {
                        for (int m = 0; m < x[n].Count; m++)
                        {
                            x[n][m] += y[n][m];
                        }
                    }
                    return x;
                });

                res = res.ConvertAll(x => x.ConvertAll(y => y));

                for (int i = 0; i < res.Count; i++)
                {
                    for (int k = 0; k < res[i].Count; k++)
                    {
                        res[i][k] = (res[i][k] / layer.Count);
                    }
                }
                dp.Add(res);
            }

            return dp;
        }
        public List<List<List<double>>> MatrixMultiplication(List<List<List<double>>> input, List<List<List<double>>> zRect)
        {
            var dz = new List<List<List<double>>>();
            for (int i = 0; i < input.Count; i++)
            {
                dz.Add(new List<List<double>>());
                for (int j = 0; j < input[i].Count; j++)
                {
                    dz[i].Add(new List<double>());
                    for (int n = 0; n < zRect[i][0].Count; n++)
                    {
                        double output = 0;
                        for (int k = 0; k < zRect[i].Count; k++)
                        {
                            output += input[i][j][k] * zRect[i][k][n];
                        }
                        dz[i][j].Add(output);
                    }
                }
            }

            return dz;
        }

        public List<List<List<double>>> DcCalc(List<List<List<double>>> input, List<List<List<double>>> dp)
        {
            var dc = new List<List<List<double>>>(input);
            dc = dc.ConvertAll(x => x.ConvertAll(y => y.ConvertAll(k => k)));
            for (int i = 0; i < dc.Count; i++)
            {
                var layer = new List<double>();
                for (int j = 0; j < dc[i].Count; j += stride)
                {
                    for (int m = 0; m < dc[i][j].Count; m += stride)
                    {
                        var x = j;
                        var y = m;
                        for (int k = 0; k < poolSize; k++)
                        {
                            for (int n = 1; n < poolSize; n++)
                            {
                                if (j + k + 1 >= dc[i].Count || m + n + 1 >= dc[i][j + k].Count)
                                {
                                    continue;
                                }
                                if (dc[i][j + k][m + n] > dc[i][x][y])
                                {
                                    x = j + k;
                                    y = m + n;
                                }
                                dc[i][j + k][m + n] = 0;
                            }
                        }
                        dc[i][x][y] = dp[i][(int)Math.Floor(((double)x) / stride)][(int)Math.Floor(((double)y) / stride)];
                    }
                }
            }

            return dc;
        }

        public List<List<List<double>>> Rotate(List<List<List<double>>> input)
        {
            var rotated = new List<List<List<double>>>(input);
            rotated = rotated.ConvertAll(x => x.ConvertAll(y => y.ConvertAll(n => n)));
            for (int i = 0; i < input.Count; i++)
            {
                for (int j = 0; j < input[i].Count; j++)
                {
                    for (int k = 0; k < input[i].Count; k++)
                    {
                        rotated[i][(j + 1) / input[i].Count][(k + 1) / input[i][j].Count] = input[i][j][k];
                    }
                }
            }
            return rotated;
        }

        public List<List<List<double>>> AddPadding(List<List<List<double>>> input)
        {
            var padding = 2 * filterSize - 2;
            var paddedZ = new List<List<List<double>>>();
            for (int j = 0; j < input.Count; j++)
            {
                paddedZ.Add(new List<List<double>>());
                for (int k = 0; k < input[j].Count + padding; k++)
                {
                    paddedZ[j].Add(new List<double>());

                    int l = (k < padding / 2 || k >= input[j].Count + 1) ? 0 : k;

                    for (int n = 0; n < input[j][Math.Abs(l - 1)].Count + padding; n++)
                    {
                        int c = n < padding / 2 || n >= input[j][Math.Abs(l - 1)].Count + 1 ? 0 : n;

                        if (l == 0 || c == 0)
                        {
                            paddedZ[j][k].Add(0);
                            continue;
                        }
                        paddedZ[j][k].Add(input[j][l - 1][c - 1]);
                    }
                }
            }

            return paddedZ;
        }

        public List<double> Flatten(List<List<List<double>>> input)
        {
            var f = new List<double>();
            for (int i = 0; i < input.Count; i++)
            {
                for (int j = 0; j < input[i].Count; j++)
                {
                    for (int k = 0; k < input[i][j].Count; k++)
                    {
                        f.Add(input[i][j][k]);
                    }
                }
            }
            return f;
        }

        public List<List<List<double>>> RectZ(List<List<List<double>>> z)
        {
            var zRect = new List<List<List<double>>>(z);
            zRect = zRect.ConvertAll(x => x.ConvertAll(y => y.ConvertAll(k => k)));

            for (int i = 0; i < zRect.Count; i++)
            {
                for (int j = 0; j < zRect[i].Count; j++)
                {
                    for (int k = 0; k < zRect[i][j].Count; k++)
                    {
                        zRect[i][j][k] = zRect[i][j][k] > 0 ? 1 : 0;
                    }
                }
            }

            return zRect;
        }

        public List<double> ConvolutionalFeedFoward(List<double> flattenInput)
        {
            var input = new List<List<double>>() { };
            for (int i = 0; i < lines; i++)
            {
                input.Add(new List<double>());
                for (int j = 0; j < columns; j++)
                {
                    input[i].Add(flattenInput[i * lines + j]);
                }
            }
            var layer = new List<List<List<double>>>() { input };
            for (int i = 0; i < filters.Count; i++)
            {
                var newLayer = new List<List<List<double>>>();
                for (int j = 0; j < layer.Count; j++)
                {
                    for (int k = 0; k < filters[i].Count; k++)
                    {
                        var z = Convolution(layer[j], filters[i][k], convolutionBiases[i][k]);
                        var c = ReLu(z);
                        var p = Pooling(c);
                        newLayer.Add(p);
                    }
                }
                layer = newLayer;
            }
            return Flatten(layer);
        }

        public List<List<double>> Convolution(List<List<double>> input, List<List<double>> filter, double? bias = null)
        {
            bias ??= 0;
            var result = new List<List<double>>();
            for (int i = 0; i < input.Count - filter.Count + 1; i++)
            {
                var dot = new List<double>();
                for (int j = 0; j < input[i].Count - filter[0].Count + 1; j++)
                {
                    double output = 0;
                    for (int k = 0; k < filter.Count; k++)
                    {
                        for (int n = 0; n < filter[k].Count; n++)
                        {
                            output += input[i + k][j + n] * filter[k][n];
                        }
                    }
                    output /= filter.Count * filter.Count;
                    dot.Add((double)(output + bias));
                }
                result.Add(dot);
            }
            return result;
        }

        public List<List<double>> Pooling(List<List<double>> input)
        {
            var result = new List<List<double>>();
            for (int i = 0; i < input.Count; i += stride)
            {
                var layer = new List<double>();
                for (int j = 0; j < input[i].Count; j += stride)
                {
                    double output = input[i][j];
                    for (int k = 0; k < poolSize; k++)
                    {
                        if (i + k >= input.Count) continue;

                        for (int n = 1; n < poolSize; n++)
                        {
                            if (j + n >= input[i].Count) continue;

                            if (input[i + k][j + n] > output) output = input[i + k][j + n];
                        }
                    }
                    layer.Add(output);
                }
                result.Add(layer);
            }
            return result;
        }

        public List<List<double>> ReLu(List<List<double>> input)
        {
            var result = new List<List<double>>(input);
            result = input.ConvertAll(x => x.ConvertAll(y => y));
            for (int i = 0; i < input.Count; i++)
            {
                for (int j = 0; j < input[i].Count; j++)
                {
                    result[i][j] = Math.Abs(input[i][j]);
                }
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
        public response Load(int take, bool print = false)
        {
            Console.WriteLine("\nBegin\n");

            FileStream ifsLabels =
             new FileStream(@"train-labels.idx1-ubyte",
             FileMode.Open); // test labels

            FileStream ifsImages =
             new FileStream(@"train-images.idx3-ubyte",
             FileMode.Open); // test images

            BinaryReader brLabels = new BinaryReader(ifsLabels);

            BinaryReader brImages = new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            if (print)
            {
                Print(brImages, brLabels);
                return new response(new List<List<double>>(), new List<List<double>>());
            }
            var results = new List<List<double>>();
            var result = new List<List<double>>();
            for (int di = 0; di < take; di++)
            {
                var pixels = new List<double>();
                for (int i = 0; i < 784; i++)
                {
                    var b = brImages.ReadByte();
                    pixels.Add(b / 255.0);
                    if (print)
                    {
                        Console.Write(b.ToString() + ", ");
                    }
                }
                result.Add(pixels);

                results.Add(Enumerable.Repeat(0.0, 10).ToList());
                results[^1][brLabels.ReadByte()] = 1;
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
            var result = loader.Load(60000);

            var net = new Network("save.txt");
            for (int i = 0; i < 1; i++)
            {
                net.StochasticGradientDescent(result.Data, result.Result, 30, 0.1, 20);
            }
        }
    }
}