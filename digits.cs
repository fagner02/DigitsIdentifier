using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace Digits
{
    class Network
    {
        public List<Int> sizes;
        public List<List<Int>> layers;
        public List<List<List<Double>>> weights;
        public List<List<List<Double>>> biases;
        public List<List<List<Double>>> gradients;
        public List<List<int>> tests;
        public List<int> tests_results;
        public List<List<int>> input;

        public Network()
        {
            sizes = new List<Int>() { 784, 50, 10 };
            layers = new List<List<Int>>() {
                new List<Int>() { 784 },
                new List<Int>() { 50 },
                new List<Int>() { 10 }
            };

            List<string> lines = File.ReadAllLines("mnist/train-images.idx3-ubyte").ToList();

            for (int i = 0; i < weights.Count(); i++)
            {
                for (int j = 0; j < sizes[i + 1]; j++)
                {
                    weights[i][j] =
                }
            }
        }

        public void Save()
        {
            string filename = "C:/Users/Fagner/Downloads/digits/save.txt";
            File.WriteAllText(filename, "");
        }

        public void FeedForward(List<Double> a)
        {
            Double sum = 0;
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i + 1].Count; j++)
                {
                    sum += layers[i + 1][j] * weights[i][j] + biases[i][j];
                }
                var newLayer = sigmoid(sum);
            }
            return newLayer;
        }

        public void Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public void Evaluate()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i + 1].Count; j++)
                {
                    layers[i + 1][j] = sigmoid(layers[i][j] * weights[i][j] + biases[i][j]);
                }
            }
        }
    }

    class Program
    {
        static void Main()
        {

        }
    }
}