export const getLondonAQIWeather = async () => {
  try {
    // Air quality from Open-Meteo
    const airUrl = `https://air-quality-api.open-meteo.com/v1/air-quality?latitude=51.5074&longitude=-0.1278&hourly=pm10,pm2_5,ozone,nitrogen_dioxide&past_days=1`;
    const airRes = await fetch(airUrl, { timeout: 20000 });
    const airData = await airRes.json();

    const now = new Date();
    const airHourIndex = airData.hourly.time.findIndex(
      (t) => new Date(t).getHours() === now.getHours()
    );

    const air = {
      pm10: airData.hourly.pm10[airHourIndex],
      pm2_5_24h_avg: airData.hourly.pm2_5[airHourIndex],
      ozone: airData.hourly.ozone[airHourIndex],
      nitrogen_dioxide: airData.hourly.nitrogen_dioxide[airHourIndex],
    };

    // Weather from OpenWeatherMap
    const apiKey = "d8d8574b09dd165fd64c98c47070d233";
    const weatherUrl = `https://api.openweathermap.org/data/2.5/forecast?lat=51.5074&lon=-0.1278&appid=${apiKey}&units=metric`;
    const weatherRes = await fetch(weatherUrl, { timeout: 20000 });
    const weatherData = await weatherRes.json();

    const weather = weatherData.list.find((entry) => {
      const entryHour = new Date(entry.dt_txt).getHours();
      return entryHour === now.getHours();
    });

    return {
      ...air,
      temp: weather.main.temp,
      humidity: weather.main.humidity,
      pressure: weather.main.pressure,
      wind_speed: weather.wind.speed,
    };
  } catch (err) {
    console.error("Failed to fetch London air or weather data:", err.message);
    return null;
  }
};
