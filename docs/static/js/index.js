$(document).ready(function() {
  // Toggle the mobile navbar menu in Bulma.
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });
});
