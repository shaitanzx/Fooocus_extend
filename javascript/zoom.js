// БЫЛО (неправильно):
const diameter = Number.isFinite(value) && value > 0 ? value : 20;
return diameter / 2;

// СТАЛО (правильно, как в оригинальном FWDF-189):
return Number.isFinite(value) && value > 0 ? value : 20;