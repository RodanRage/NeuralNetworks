using System.Diagnostics;

namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Random rand = new Random();

            float[][] weights0_1 = new float[][]
            {
                new float[] {rand.NextSingle(), rand.NextSingle(), rand.NextSingle() },
                new float[] {rand.NextSingle(), rand.NextSingle(), rand.NextSingle() },
                new float[] {rand.NextSingle(), rand.NextSingle(), rand.NextSingle() },
                new float[] {rand.NextSingle(), rand.NextSingle(), rand.NextSingle() }
            };
            float[] weights1_2 = new float[] 
            { 
                rand.NextSingle(), 
                rand.NextSingle(), 
                rand.NextSingle(), 
                rand.NextSingle() 
            };
            float alpha = 0.2f;

            byte[][] streetLights = new byte[][] 
            { 
                new byte[] { 1, 0, 1 }, 
                new byte[] { 0, 1, 1 }, 
                new byte[] { 0, 0, 1 },
                new byte[] { 1, 1, 1 }
            };
            byte[,] walkStop = new byte[4, 1]
            {
                { 1 },
                { 1 },
                { 0 },
                { 0 }
            };

            byte[] layer0 = streetLights[0];
            
            float[] layer1 = new float[4];
            for (int i = 0; i < layer1.Length; i++)
            {
                layer1[i] = Relu(MatrixOperations.GetWsum(layer0, weights0_1[i]));
            }
            
            float[] layer2 = new float[2];
            for (int i = 0; i < layer2.Length; i++)
            {
                layer2[i] = Relu(MatrixOperations.GetWsum(layer1, weights1_2));
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

        static float Relu(float x)
        {
            if (x > 0) return x;
            else return 0;
        }
        
    }
}
