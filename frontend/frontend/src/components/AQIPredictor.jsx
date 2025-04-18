import { useState, useEffect, useMemo } from "react";
import { predictAQI } from "../utils/api";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LineController,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  PointElement,
  LineController,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

import Holidays from "date-holidays";
import { getLondonAQIWeather } from "../utils/getRealTimeLondonData";

const AQIPredictor = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const now = new Date();
      const hd = new Holidays("GB"); // UK holidays
      const isHoliday = hd.isHoliday(now) ? 1 : 0;
      const isWeekend = now.getDay() >= 5 ? 1 : 0;

      const liveData = await getLondonAQIWeather();
      if (!liveData) throw new Error("Live data not available.");

      const predictionData = {
        ...liveData,
        pm10_24h_avg: liveData.pm10,
        hour_sin: Math.sin((2 * Math.PI * now.getHours()) / 24),
        hour_cos: Math.cos((2 * Math.PI * now.getHours()) / 24),
        month_sin: Math.sin((2 * Math.PI * (now.getMonth() + 1)) / 12),
        month_cos: Math.cos((2 * Math.PI * (now.getMonth() + 1)) / 12),
        day_of_week: now.getDay(),
        is_holiday: isHoliday,
        is_weekend: isWeekend,
      };

      const { data } = await predictAQI(predictionData);
      setResult(data);
    } catch (error) {
      console.error("Full error:", {
        message: error.message,
        response: error.response?.data,
      });
      alert(
        `Prediction failed: ${error.response?.data?.error || error.message}`
      );
    }
    setLoading(false);
  };

  const getAqiStatus = (aqi) => {
    if (!aqi) return { level: "--", emoji: "😶", color: "gray" };
    if (aqi <= 50) return { level: "Very Good", emoji: "😊", color: "green" };
    if (aqi <= 100) return { level: "Moderate", emoji: "😐", color: "yellow" };
    if (aqi <= 150) return { level: "Unhealthy", emoji: "😷", color: "orange" };
    return { level: "Hazardous", emoji: "😵", color: "red" };
  };

  const aqiStatus = getAqiStatus(result?.predicted_values?.pm2_5);

  const chartData = useMemo(() => {
    if (!result) return { labels: [], datasets: [] };

    return {
      labels: result.chart_data.labels,
      datasets: [
        {
          label: "PM2.5",
          data: result.chart_data.pm2_5,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.2)",
          tension: 0.4,
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          label: "PM10",
          data: result.chart_data.pm10,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.4,
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    };
  }, [result]);

  const getGradientPosition = (aqi) => {
    if (!aqi) return 20;
    if (aqi <= 50) return 20;
    if (aqi <= 100) return 40;
    if (aqi <= 150) return 60;
    return 80;
  };

  const currentHourIndex = currentTime.getHours();

  return (
    <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg overflow-hidden font-sans">
      <div className="p-6">
        <div className="flex items-center mb-2">
          <div
            className={`bg-${aqiStatus.color}-100 text-${aqiStatus.color}-800 rounded-full px-3 py-1 flex items-center`}
          >
            <span className="text-xl mr-1">{aqiStatus.emoji}</span>
            <span className="font-medium">{aqiStatus.level}</span>
          </div>
        </div>

        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800">London, UK</h1>
          <div className="text-gray-600">United Kingdom</div>
        </div>

        <div className="absolute top-4 right-4">
          <div className="relative">
            <svg width="100" height="50" viewBox="0 0 100 50">
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="50%" stopColor="#fbbf24" />
                  <stop offset="100%" stopColor="#ef4444" />
                </linearGradient>
              </defs>
              <path
                d="M 10,40 A 40,40 0 0 1 90,40"
                stroke="url(#gradient)"
                strokeWidth="8"
                fill="none"
                strokeLinecap="round"
              />
              <circle
                cx={
                  10 +
                  getGradientPosition(result?.predicted_values?.pm2_5) * 0.8
                }
                cy="40"
                r="6"
                fill="white"
                stroke="#10b981"
                strokeWidth="3"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="text-3xl font-bold mt-2">
                {result?.predicted_values?.pm2_5?.toFixed(0) || "--"}
              </div>
              <div className="text-xs">AQI</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-3 mb-6">
          <div className="bg-gray-50 p-3 rounded-lg shadow-sm">
            <div className="text-sm text-gray-500 mb-1">PM2.5</div>
            <div className="text-xl font-bold">
              {result?.predicted_values?.pm2_5?.toFixed(2) || "--"}
            </div>
            <div className="text-xs text-gray-400">µg/m³</div>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg shadow-sm">
            <div className="text-sm text-gray-500 mb-1">PM10</div>
            <div className="text-xl font-bold">
              {result?.predicted_values?.pm10?.toFixed(2) || "--"}
            </div>
            <div className="text-xs text-gray-400">µg/m³</div>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg shadow-sm">
            <div className="text-sm text-gray-500 mb-1">NO2</div>
            <div className="text-xl font-bold">
              {result?.predicted_values?.no2?.toFixed(2) || "--"}
            </div>
            <div className="text-xs text-gray-400">µg/m³</div>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg shadow-sm">
            <div className="text-sm text-gray-500 mb-1">O3</div>
            <div className="text-xl font-bold">
              {result?.predicted_values?.o3?.toFixed(2) || "--"}
            </div>
            <div className="text-xs text-gray-400">µg/m³</div>
          </div>
        </div>

        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full mt-2 bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 disabled:bg-blue-300 transition-colors flex items-center justify-center"
        >
          {loading ? (
            <>
              <svg
                className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                ></path>
              </svg>
              Predicting...
            </>
          ) : (
            "Predict Air Quality"
          )}
        </button>

        <div className="mb-6 mt-6">
          <h2 className="text-lg font-semibold mb-3">Air Quality Forecast</h2>
          <div className="relative h-48 mb-4 bg-white rounded-lg p-2 shadow-sm">
            {result ? (
              <Line
                key={JSON.stringify(chartData.labels)}
                data={chartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: "top",
                      labels: {
                        boxWidth: 12,
                        usePointStyle: true,
                        pointStyle: "circle",
                      },
                    },
                    tooltip: {
                      mode: "index",
                      intersect: false,
                      backgroundColor: "rgba(255, 255, 255, 0.9)",
                      titleColor: "#334155",
                      bodyColor: "#334155",
                      borderColor: "#e2e8f0",
                      borderWidth: 1,
                      padding: 10,
                      displayColors: true,
                    },
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      grid: { color: "rgba(0,0,0,0.05)", drawBorder: false },
                      ticks: { font: { size: 10 } },
                      title: {
                        display: true,
                        text: "µg/m³",
                        font: { size: 10 },
                      },
                    },
                    x: {
                      grid: { display: false },
                      ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8,
                        font: { size: 10 },
                      },
                    },
                  },
                  elements: {
                    point: {
                      radius: (ctx) =>
                        ctx.dataIndex === currentHourIndex ? 4 : 0,
                      backgroundColor: (ctx) =>
                        ctx.dataIndex === currentHourIndex
                          ? "white"
                          : ctx.dataset.borderColor,
                      borderColor: (ctx) =>
                        ctx.dataIndex === currentHourIndex
                          ? ctx.dataset.borderColor
                          : "transparent",
                      borderWidth: 2,
                      hoverRadius: 5,
                    },
                  },
                }}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <span>Predict air quality to see forecast</span>
              </div>
            )}
          </div>
        </div>

        {result && (
          <div className="bg-gray-50 rounded-lg p-4 mb-6 shadow-sm">
            <h3 className="font-semibold mb-2">Model Performance</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white p-3 rounded shadow-sm">
                <div className="text-sm text-gray-500">Accuracy</div>
                <div className="font-bold text-lg">
                  {(result.model_metrics.r2 * 100).toFixed(0)}%
                </div>
              </div>
              <div className="bg-white p-3 rounded shadow-sm">
                <div className="text-sm text-gray-500">Last Updated</div>
                <div className="font-bold text-lg">Just now</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AQIPredictor;
