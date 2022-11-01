#pragma once

class NaiveBaysian
{
private:

	size_t classesCount;
	size_t attributeValueSize;
	size_t attributeCount;
	std::vector<double> classOccurancesInTrainingData;
	std::vector<std::vector<double>> probabilityTables;

public:

	NaiveBaysian(size_t classesCountIn, size_t attributeCountIn, size_t attributeValueSizeIn);

	void train();

	std::vector<size_t> predict(std::vector<size_t> fetures)

}

void NaiveBaysian::train(std::vector<std::vector<size_t>> attributes, std::vector<size_t> classes)
{
	for(const std::vector<size_t>& instanceAttributes : attributes)
	{
		assert(attributes.size() == attributeCount);
		for(size_t attribute : instanceAttributes)
			assert(attribute < attributeValueSize);
	}
	for(size_t classNum : classes)
		assert(classNum < classesCount);

	probabilityTables.resize(attributeCount * classesCount);
	for(std::vector<double>& table : probabilityTables)
		table.resize(attributeValueSize, 0);

	//Calculate probaility table
	for(size_t i = 0; i <= attributes.size(); ++i)
	{
		getline(trainingDataFile, Buf);
		std::stringstream lineStream(Buf);

		++classOccurancesInTrainingData[classes[i]];

		for(size_t j = 0; j < attributeCount; ++j)
			probabilityTables[j*classesCount+classes[i]][attributes[j]]++;
	}

	for(size_t t = 0; t < attributeCount; ++t)
	{
		for(int d = 0; d < classesCount; ++d) {
			int correction = 0;
			// this loop judges weather there is zero occurence of some conjuction
			// if it dose, then do Laplacian correction
			for(int o = 0; o < attributeValueSize; ++o)
			{
				if(probabilityTables[(t * classesCount + d)][o] == 0)
				{
					correction = attributeValueSize;
					for(int p = 0; p < attributeValueSize; ++p)
						probabilityTables[(t * classesCount + d)][p]++;
					break;
				}
			}

			for(int w = 0; w < attributeValueSize; ++w)
			// claculate every conjuction's contribution of probability
			{
				probabilityTables[(t * classesCount + d)][w] /=
					(static_cast<size_t>(classOccurancesInTrainingData[d]) + correction);
			}
		}
	}

	// calculate the probability of each resulting class
	for (int probIndex = 0; probIndex < classesCount; ++probIndex)
		classOccurancesInTrainingData[probIndex] =
			classOccurancesInTrainingData[probIndex] / attributes.size();
}
