#include "element.h"

std::string Element::getString() const
{
	switch(type)
	{
		case E_TYPE_R:
			return "r";
		case E_TYPE_C:
			return "c";
		case E_TYPE_L:
			return "l";
		case E_TYPE_P:
			return "p";
		case E_TYPE_W:
			return "w";
		case E_TYPE_SOURCE:
			return "s";
		case E_TYPE_UNKOWN:
		default:
			return "x";
	}
}

Element::Element(ElementType typeI, cv::Rect rectI, float probI): type(typeI), rect(rectI), prob(probI)
{

}

ElementType Element::getType() const
{
	return type;
}

cv::Rect Element::getRect() const
{
	return rect;
}

double Element::getProb() const
{
	return prob;
}
