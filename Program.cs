using System.Diagnostics;

namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            float[][] weights = new float[][]
            {
                new float[]{0.1f, 0.1f, -0.3f},
                new float[] {0.1f, 0.2f, 0.0f},
                new float[] {0.0f, 1.3f, 0.1f}
            };
            
            float[] toes = new float[] { 8.5f, 9.5f, 9.9f, 9.0f };
            float[] wlrec = new float[] { 0.65f, 1.0f, 1.0f, 0.9f };
            float[] nfans = new float[] { 1.2f, 1.3f, 0.5f, 1.0f };

            float[] hurt = new float[] { 0.1f, 0.0f, 0.0f, 0.1f };
            float[] win = new float[] { 1, 1, 0, 1 };
            float[] sad = new float[] { 0.1f, 0.0f, 0.1f, 0.2f };
            
            float alpha = 0.01f;
            float[] error = new float[] { 0, 0, 0 };
            float[] delta = new float[] { 0, 0, 0 };
            float[][] weightDeltas;

            float[] input = new float[] { toes[0], wlrec[0], nfans[0] };
            float[] goalPredictions = new float[] { hurt[0], win[0], sad[0] };

            float[] prediction = GetPrediction(input, weights);
            
            for (int i = 0; i < goalPredictions.Length; i++)
            {
                delta[i] = prediction[i] - goalPredictions[i];
                error[i] = delta[i] * delta[i];
            }
           
            weightDeltas = GetOuterProd(input, delta);
            
            for (int i = 0; i < weights.Length; i++)
                for(int j = 0; j < weights[i].Length; j++)
                    weights[i][j] -= alpha * weightDeltas[i][j];

            Console.WriteLine("Weights: ");
            foreach (var weightLine in weights)
                foreach(var weight in weightLine)
                    Console.WriteLine(weight);

            Console.WriteLine("Weight Deltas: ");
            foreach (var weightDeltaLine in weightDeltas)
                foreach (var weightDelta in weightDeltaLine)
                    Console.WriteLine(weightDelta);
        }

        static float[] GetPrediction(float[] input, float[][] weights)
        {
            float[] prediction = GetVectMatMul(input, weights);
            return prediction;
        }

        static float[] GetVectMatMul(float[] vector, float[][] matrix)
        {
            float[] outputVector = new float[] { 0, 0, 0 };
            if (vector.Length != matrix.Length)
            {
                throw new Exception("Размерность массивов не совпадает");
            }
            else
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    outputVector[i] = GetWsum(vector, matrix[i]);
                }
                return outputVector;
            }
        }

        static float GetWsum(float[] firstVector, float[] secondVector)
        {
            if (secondVector.Length != firstVector.Length)
            {
                throw new Exception("Размерность массивов не совпадает");
            }
            else
            {
                float output = 0;
                for (int i = 0; i < firstVector.Length; i++)
                {
                    output += firstVector[i] * secondVector[i];
                }
                return output;
            }
        }

        static float[] GetEleMul(float scalar, float[] vector)
        {
            float[] output = new float[] { 0, 0, 0 };
            if (vector.Length == output.Length)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    output[i] = vector[i]*scalar;
                }
                return output;
            }
            throw new Exception("Несоответствие размерностей входного и выходного массивов");
        }

        static float[][] GetOuterProd(float[] firstVector, float[] secondVector)
        {
            float[][] outputMatrix = new float[firstVector.Length][];
            for(int i = 0; i < firstVector.Length; i++)
            {
                outputMatrix[i] = new float[secondVector.Length];
                for (int j = 0; j < secondVector.Length; j++)
                {
                    outputMatrix[i][j] = firstVector[j] * secondVector[i];
                }
            }
            return outputMatrix;
        }
        
    }
}
