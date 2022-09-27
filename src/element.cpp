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
		case E_TYPE_NODE:
			return "";
		case E_TYPE_COMPOSIT:
			return string.empty() ? "x" : string;
		case E_TYPE_UNKOWN:
		default:
			return "x";
	}
}

ElementType Element::typeFromChar(char c)
{
	switch(c)
	{
		case 'r':
			return E_TYPE_R;
		case 'c':
			return E_TYPE_C;
		case 'l':
			return E_TYPE_L;
		case 'p':
			return E_TYPE_P;
		case 'w':
			return E_TYPE_W;
		case 's':
			return E_TYPE_SOURCE;
		case 'x':
			return E_TYPE_COMPOSIT;
		case E_TYPE_UNKOWN:
		default:
			return E_TYPE_UNKOWN;
	}
}

int Element::getPrecedence(ElementType type)
{
	switch(type)
	{
		case E_TYPE_R:
			return 1;
		case E_TYPE_C:
			return 2;
		case E_TYPE_L:
			return 3;
		case E_TYPE_P:
			return 4;
		case E_TYPE_W:
			return 5;
		case E_TYPE_SOURCE:
			return -1;
		case E_TYPE_NODE:
			return -1;
		case E_TYPE_COMPOSIT:
			return 0;
		case E_TYPE_UNKOWN:
		default:
			return 0;
	}
}

int Element::getPrecedence() const
{
	if(type != E_TYPE_COMPOSIT || string.empty())
	{
		return getPrecedence(type);
	}
	else
	{
		int accum = 0;
		for(size_t i = 0; i < string.size(); ++i)
			accum += getPrecedence(typeFromChar(string[i]));
		return accum;
	}
}

Element::Element(ElementType typeI, cv::Rect rectI, float probI): type(typeI), rect(rectI), prob(probI)
{

}

Element::Element(const Element& a, const Element& b, bool serial)
{
	prob = a.prob + b.prob/2;
	rect.x = std::min(a.rect.x, b.rect.x);
	rect.y = std::min(a.rect.y, b.rect.y);
	int right = std::max(a.rect.x+a.rect.width, b.rect.x+b.rect.width);
	int bottom = std::max(a.rect.y+a.rect.height, b.rect.y+b.rect.height);

	rect.width = right-rect.x;
	rect.height = bottom-rect.y;

	type = E_TYPE_COMPOSIT;

	if(serial)
	{
		if(a.getPrecedence() < b.getPrecedence())
			string = a.getString() + "-" + b.getString();
		else
			string = b.getString() + "-" + a.getString();
	}
	else
	{
		if(a.getPrecedence() < b.getPrecedence())
			string = std::string("(") + a.getString() + std::string(")(") + b.getString() + ")";
		else
			string = std::string("(") + b.getString() + std::string(")(") + a.getString() + ")";
	}
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

cv::Point Element::center() const
{
	cv::Rect rect = getRect();
	return cv::Point(rect.x+rect.width/2, rect.y+rect.height/2);
}
