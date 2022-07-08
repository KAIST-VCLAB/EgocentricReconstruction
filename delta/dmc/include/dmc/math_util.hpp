#pragma once
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace dmc
{
	template <class T>
	auto squared(const T& t)
	{
		return t * t;
	}

	template <class T>
	auto sign(const T& t)
	{
		if (t < static_cast<T>(0))
			return static_cast<T>(-1);

		if (t > static_cast<T>(0))
			return static_cast<T>(1);

		return static_cast<T>(0);
	}
	template <class T, class U>
	auto lerp(T x1, T x2, U ratio)
	{
		return x1 + (x2 - x1) * ratio;
	}

	template <class T>
	auto invlerp(T x1, T x2, T x)
	{
		return (x - x1) / (x2 - x1);
	}
}
