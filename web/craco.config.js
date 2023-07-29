module.exports = {
    jest: {
        configure: {
            testEnvironment: "jsdom",
            testMatch: [
                "<rootDir>/src/**/__tests__/**/*.{js,jsx,ts,tsx}",
                "<rootDir>/src/**/*.{spec,test}.{js,jsx,ts,tsx}",
                "<rootDir>/test/**/*.{js,jsx,ts,tsx}"
            ]
        },
    },
};