using System.Diagnostics;

namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
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

            float[] weights = new float[] { 0.5f, 0.48f, -0.7f };
            float alpha = 0.1f;
            byte[] input = streetLights[0];
            byte goalPrediction = walkStop[0, 0];
            float prediction, error, delta;

            for (int i = 0; i < 20; i++)
            {
                prediction = GetPrediction(input, weights);
                delta = prediction - goalPrediction;
                error = delta * delta;
                
                for(int j = 0; j < weights.Length; j++)
                {
                    weights[j] = weights[j] - (alpha * input[j] * delta);
                }
                Console.WriteLine($"{i}) Error: {error}, Prediction: {prediction}");
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
