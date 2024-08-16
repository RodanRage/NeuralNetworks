using System.Diagnostics;

namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            float[] weights = new float[] { 0.5f, 0.48f, -0.7f };
            float alpha = 0.1f;

            byte[][] streetLights = new byte[][] 
            { 
                new byte[] { 1, 0, 1 }, 
                new byte[] { 0, 1, 1 }, 
                new byte[] { 0, 0, 1 },
                new byte[] { 1, 1, 1 },
                new byte[] { 0, 1, 1 },
                new byte[] { 1, 0, 1 }
            };
            byte[,] walkStop = new byte[6, 1]
            {
                { 0 },
                { 1 },
                { 0 },
                { 1 },
                { 1 },
                { 0 }
            };


            byte[] input;
            byte goalPrediction;
            float prediction, error, delta, errorForAllLights;

            for (int i = 0; i < 40; i++)
            {
                errorForAllLights = 0;
                for (int k = 0; k < walkStop.Length; k++)
                {
                    input = streetLights[k];
                    goalPrediction = walkStop[k, 0];

                    prediction = GetPrediction(input, weights);
                    delta = prediction - goalPrediction;
                    error = delta * delta;
                    errorForAllLights += error;

                    for (int j = 0; j < weights.Length; j++)
                    {
                        weights[j] = weights[j] - (alpha * input[j] * delta);
                    }
                    Console.WriteLine($"k = {k}) Prediction: {prediction}");
                }
                Console.WriteLine($"{i}) Error: {errorForAllLights}");
            }
        }

        static float GetPrediction(byte[] input, float[] weights)
        {
            float prediction = MatrixOperations.GetWsum(input, weights);
            return prediction;
        }

        static float[] GetPrediction(float[] input, float[][] weights)
        {
            float[] prediction = MatrixOperations.GetVectMatMul(input, weights);
            return prediction;
        }

        
        
    }
}
