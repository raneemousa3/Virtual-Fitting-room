/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/static/js/**/*.{js,jsx,ts,tsx}",
    "./app/static/index.html",
  ],
  theme: {
    extend: {
      colors: {
        gold: '#C8A165',
        ivory: '#F5F5F0',
      },
      fontFamily: {
        serif: ['Playfair Display', 'serif'],
        sans: ['Montserrat', 'sans-serif'],
      },
    },
  },
  plugins: [],
} 