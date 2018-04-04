#include "cntkbrain.hpp"

#include <CNTKLibrary.h>

#include <iostream>

namespace Aronda::Trainer
{

namespace
{

    using ElemType = float;

    auto FullyConnectedLinearLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device,
                                   const std::wstring& outputName = L"", unsigned long seed = 1)
    {
        assert(input.Shape().Rank() == 1);
        size_t inputDim = input.Shape()[0];

        auto timesParam = CNTK::Parameter(
            {outputDim, inputDim}, CNTK::DataType::Float,
            CNTK::GlorotUniformInitializer(CNTK::DefaultParamInitScale, CNTK::SentinelValueForInferParamInitRank,
                                           CNTK::SentinelValueForInferParamInitRank, seed),
            device, L"timesParam");
        auto timesFunction = Times(timesParam, input, L"times");

        auto plusParam = CNTK::Parameter({outputDim}, 0.0f, device, L"plusParam");
        return Plus(plusParam, timesFunction, outputName);
    }

    auto FullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device,
                                const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity,
                                const std::wstring& outputName = L"", unsigned long seed = 1)
    {
        return nonLinearity(FullyConnectedLinearLayer(input, outputDim, device, outputName, seed));
    }

    auto FullyConnectedFeedForwardClassifierNet(
        CNTK::Variable input, size_t numOutputClasses, size_t hiddenLayerDim, size_t numHiddenLayers,
        const CNTK::DeviceDescriptor& device,
        const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity, const std::wstring& outputName,
        unsigned long seed = 1)
    {
        assert(numHiddenLayers >= 1);
        auto classifierRoot = FullyConnectedDNNLayer(input, hiddenLayerDim, device, nonLinearity, L"", seed);
        for(size_t i = 1; i < numHiddenLayers; ++i)
            classifierRoot = FullyConnectedDNNLayer(classifierRoot, hiddenLayerDim, device, nonLinearity, L"", seed);

        auto outputTimesParam = CNTK::Parameter({numOutputClasses, hiddenLayerDim}, CNTK::DataType::Float,
                                                CNTK::UniformInitializer(0.5, seed), device);
        return Times(outputTimesParam, classifierRoot, 1, outputName);
    }

    void PrintTrainingProgress(const CNTK::TrainerPtr trainer, size_t minibatchIdx, size_t outputFrequencyInMinibatches)
    {
        if((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer->PreviousMinibatchSampleCount() != 0)
        {
            double trainLossValue = trainer->PreviousMinibatchLossAverage();
            double evaluationValue = trainer->PreviousMinibatchEvaluationAverage();
            // std::cout << "Minibatch " << minibatchIdx << ": loss = " << trainLossValue
            //           << ", Evaluation criterion = " << evaluationValue << std::endl;
        }
    }

    inline bool getVariableByName(std::vector<CNTK::Variable> variableLists, std::wstring varName, CNTK::Variable& var)
    {
        for(std::vector<CNTK::Variable>::iterator it = variableLists.begin(); it != variableLists.end(); ++it)
        {
            if(it->Name().compare(varName) == 0)
            {
                var = *it;
                return true;
            }
        }
        return false;
    }

    inline bool getInputVariableByName(CNTK::FunctionPtr evalFunc, std::wstring varName, CNTK::Variable& var)
    {
        return getVariableByName(evalFunc->Arguments(), varName, var);
    }

    inline bool getOutputVariableByName(CNTK::FunctionPtr evalFunc, std::wstring varName, CNTK::Variable& var)
    {
        return getVariableByName(evalFunc->Outputs(), varName, var);
    }
}

namespace Impl
{
    class CntkBrain
    {

        static const std::wstring c_input_var_name;
        static const std::wstring c_output_var_name;
        static const std::wstring c_labels_var_name;

    public:
        CntkBrain()
            : m_device(CNTK::DeviceDescriptor::UseDefaultDevice())
        {
            CNTK::NDShape shape{Aronda::State::number_of_state_per_square * Aronda::State::number_of_square};
            m_input = CNTK::InputVariable(shape, CNTK::DataType::Float, c_input_var_name);
            m_model = FullyConnectedFeedForwardClassifierNet(m_input, Aronda::State::number_of_square, 2048, 10, m_device,
                                                             std::bind(CNTK::Sigmoid, std::placeholders::_1, L""),
                                                             c_output_var_name);
            getOutputVariableByName(m_model, c_output_var_name, m_output);
            m_labels = CNTK::InputVariable({ m_output.Shape().TotalSize() }, CNTK::DataType::Float, c_labels_var_name);

            auto trainingLoss = SquaredError(m_model, m_labels, L"lossFunction");
            auto prediction = SquaredError(m_model, m_labels, L"predictionError");

            // python example
            // # loss = 'mse'
            // loss = C.reduce_mean(C.square(model - q_target), axis = 0)
            // meas = C.reduce_mean(C.square(model - q_target), axis = 0)
            //
            // # optimizer
            // lr = 0.00025
            // lr_schedule = C.learning_parameter_schedule(lr)
            // learner = C.sgd(model.parameters, lr_schedule, gradient_clipping_threshold_per_sample = 10)
            // trainer = C.Trainer(model, (loss, meas), learner)

            CNTK::LearningRateSchedule learningRatePerSample = 0.01;
            m_trainer = CNTK::CreateTrainer(m_model, trainingLoss, prediction,
                                            {CNTK::SGDLearner(m_model->Parameters(), learningRatePerSample)});
        }

        void save(const std::string& path) const
        {
            m_model->Save({path.begin(), path.end()});
        }

        Action predict(const State& current_state) const
        {
            const std::size_t number_of_samples = 1; // batch of 1

            std::vector<ElemType> inputs(current_state.size());
            for(int i = 0; i < current_state.size(); i++)
                inputs[i] = current_state(i);
            CNTK::NDShape inputShape = m_input.Shape().AppendShape({1, number_of_samples});
            CNTK::ValuePtr inputValue = CNTK::MakeSharedObject<CNTK::Value>(
                CNTK::MakeSharedObject<CNTK::NDArrayView>(inputShape, inputs, true));

            CNTK::ValuePtr outputValue;
            std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{m_output, outputValue}};
            m_model->Evaluate({{m_input, inputValue}}, outputs, m_device);

            outputValue = outputs[m_output];
            CNTK::NDShape outputShape = m_output.Shape().AppendShape({1, number_of_samples});
            std::vector<ElemType> outputData(outputShape.TotalSize());
            CNTK::NDArrayViewPtr cpuArrayOutput
                = CNTK::MakeSharedObject<CNTK::NDArrayView>(outputShape, outputData, false);
            cpuArrayOutput->CopyFrom(*outputValue->Data());

            Action res;
            for(int i = 0; i < Aronda::State::number_of_square; i++)
                res[i] = outputData[i];
            return res;
        }

        /**
        * @pre states.size() == actions.size()
        */
        void train(const std::vector<State>& states, const std::vector<Action>& actions)
        {
            Expects(states.size() == actions.size());
            const size_t outputFrequencyInMinibatches = 1;
            const auto minibatch = createMiniBatch(states, actions);
            m_trainer->TrainMinibatch({{m_input, minibatch.first}, {m_labels, minibatch.second}}, m_device);
            PrintTrainingProgress(m_trainer, 0, outputFrequencyInMinibatches);
        }

    private:
        const CNTK::DeviceDescriptor m_device;
        CNTK::FunctionPtr m_model;
        CNTK::Variable m_input;
        CNTK::Variable m_output;
        CNTK::Variable m_labels;
        CNTK::TrainerPtr m_trainer;

        /**
        * @pre states.size() == actions.size()
        */
        std::pair<CNTK::ValuePtr, CNTK::ValuePtr> createMiniBatch(const std::vector<State>& states, const std::vector<Action>& actions) const
        {
            Expects(states.size() == actions.size());
            const std::size_t batchSize = states.size();
            std::vector<float> inputData;
            inputData.reserve(states.front().size() * batchSize);
            std::vector<float> outputData;
            outputData.reserve(actions.front().size() * batchSize);

            for (std::size_t b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < states[b].size(); i++)
                {
                    inputData.push_back(states[b](i));
                }
                for (int n = 0; n < actions[b].size(); n++)
                {
                    outputData.push_back(actions[b](n));
                }
            }

            auto inputDataValue = CNTK::Value::CreateBatch(m_input.Shape(), inputData, m_device, true);
            auto outputDataValue = CNTK::Value::CreateBatch(m_output.Shape(), outputData, m_device, true);
            return std::make_pair(inputDataValue, outputDataValue);
        }
    };

    const std::wstring CntkBrain::c_input_var_name = L"state";
    const std::wstring CntkBrain::c_output_var_name = L"q_value";
    const std::wstring CntkBrain::c_labels_var_name = L"labels";
}

CntkBrain::CntkBrain()
    : m_pimpl(std::make_unique<Impl::CntkBrain>())
{
}

CntkBrain::~CntkBrain() = default;

void CntkBrain::impl_save(const std::string& path) const
{
    m_pimpl->save(path);
}

Action CntkBrain::impl_predict(const State& current_state) const
{
    return m_pimpl->predict(current_state);
}

void CntkBrain::impl_train(const std::vector<State>& states, const std::vector<Action>& actions)
{
    m_pimpl->train(states, actions);
}
}
