namespace NeuralNetworks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            float[] weights = new float[] { 0.3f, 0.2f, 0.9f };
            float[] wlrec = new float[] { 0.65f, 1.0f, 1.0f, 0.9f };
            float[] hurt = new float[] { 0.1f, 0.0f, 0.0f, 0.1f };
            float[] win = new float[] { 1, 1, 0, 1 };
            float[] sad = new float[] { 0.1f, 0.0f, 0.1f, 0.2f };
            float[] error = new float[] { 0, 0, 0 };
            float[] delta = new float[] { 0, 0, 0 };
            float[] weightDeltas;
            float alpha = 0.1f;

            float input = wlrec[0];
            float[] goalPredictions = new float[] { hurt[0], win[0], sad[0] };

            float[] prediction = GetPrediction(input, weights);
            
            for (int i = 0; i < goalPredictions.Length; i++)
            {
                delta[i] = prediction[i] - goalPredictions[i];
                error[i] = delta[i] * delta[i];
            }

            weightDeltas = EleMul(input, delta);
            
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] -= alpha * weightDeltas[i];
            }

            Console.WriteLine("Weights: ");
            foreach (var weight in weights)
                Console.WriteLine(weight);

            Console.WriteLine("Weight Deltas: ");
            foreach (var weightDelta in weightDeltas)
                Console.WriteLine(weightDelta);
        }

        static float[] GetPrediction(float input, float[] weights)
        {
            float[] prediction = EleMul(input, weights);
            return prediction;
        }

        static float[] EleMul(float input, float[] weights)
        {
            float[] output = new float[] { 0, 0, 0 };
            if (weights.Length == output.Length)
            {
                for (int i = 0; i < weights.Length; i++)
                {
                    output[i] = weights[i]*input;
                }
                return output;
            }
            throw new Exception("Несоответствие размерностей входного и выходного массивов");
        }
    }
}
