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
            float[,] walkStop = new float[4, 1]
            {
                { 1 },
                { 1 },
                { 0 },
                { 0 }
            };

            byte[] layer0;
            float[] layer1 = new float[4];
            float layer2;
            float layer2Delta;

            for (int i = 0; i < 60; i++)
            {
                float layer2Error = 0;
                for (int j = 0; j < streetLights.Length; j++)
                {
                    layer0 = streetLights[j];
                    for (int k = 0; k < layer1.Length; k++)
                    {
                        layer1[k] = Relu(MatrixOperations.GetWsum(layer0, weights0_1[k]));
                    }
                    
                    layer2 = Relu(MatrixOperations.GetWsum(layer1, weights1_2));

                    layer2Delta = layer2 - walkStop[i, 0];
                    layer2Error = layer2Delta * layer2Delta; 
                }
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

        static float ReluToDeriv(float x)
        {
            if (x > 0) return 1;
            else return 0;
        }
    }
}
