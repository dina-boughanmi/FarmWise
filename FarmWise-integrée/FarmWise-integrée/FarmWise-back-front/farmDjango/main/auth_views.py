from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse  # for safe URL building
import stripe  # or other payment gateway



def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        plan = request.POST.get('plan')  # Get the selected plan

        # Debugging line
        print(f"Selected plan: {plan}")

        # Basic validation
        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('signup')

        # Create user
        user = User.objects.create_user(username=username, email=email, password=password1)

        # Optional: Store plan in session or user profile
        request.session['selected_plan'] = plan  # Save plan for payment step

        login(request, user)

        # Debugging line
        print(f"Redirecting to payment with plan: {plan}")

        # Redirect to payment with selected plan
        return redirect(reverse('payment') + f'?plan={plan}')

    return render(request, 'main/signUp.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, "Invalid username or password.")
            return redirect('login')

    return render(request, 'main/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib import messages


@login_required
def profile_view(request):
    user = request.user
    plan = request.session.get('selected_plan', 'No plan selected')  # Default to 'No plan selected' if no plan is found

    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST.get('password')

        # Check if username or email changed and are valid
        if username != user.username and User.objects.filter(username=username).exclude(id=user.id).exists():
            messages.error(request, "Username already taken.")
            return redirect('profile')

        if email != user.email and User.objects.filter(email=email).exclude(id=user.id).exists():
            messages.error(request, "Email already taken.")
            return redirect('profile')

        # Save updates
        user.username = username
        user.email = email
        if password:
            user.set_password(password)  # hash new password
        user.save()

        messages.success(request, "Profile updated successfully.")
        if password:
            # re-authenticate after password change
            login(request, user)
        return redirect('profile')
    return render(request, 'main/profile.html', {'plan': plan})

stripe.api_key = "sk_test_51OqJbyP8fZ7mtZrf1ucCZuWgiG9TwaEZeDfFgVTC7dM2lapkIL1Hiq8LJKjLK2YVbdlYTFCHI3FgIthXHJgTxaa800QGib2xV2"

def payment_view(request):
    # If POST, assume plan should already be in session
    if request.method == "POST":
        plan = request.session.get('selected_plan')  # Only from session during POST
    else:
        # On GET: try to fetch from URL first, then session
        plan = request.GET.get('plan')
        if plan:
            request.session['selected_plan'] = plan  # Save it in session
        else:
            plan = request.session.get('selected_plan')  # Fallback if not in GET

    print(f"DEBUG: plan = {plan}")

    if not plan:
        messages.error(request, "No plan selected.")
        return redirect('signup')

    if request.method == "POST":
        token = request.POST.get('stripeToken')
        if not token:
            messages.error(request, "Stripe token missing.")
            return render(request, 'main/payment.html', {'plan': plan})

        try:
            # Example: Replace with your actual plan-to-price logic
            amount = calculate_price(plan)

            charge = stripe.Charge.create(
                amount=amount,
                currency="USD",
                source=token,
                description=f"Payment for {plan}",
            )

            messages.success(request, "Payment successful! Your subscription has been activated.")
            return render(request, 'main/payment.html', {'plan': plan})

        except stripe.error.StripeError as e:
            messages.error(request, f"Payment failed: {e.user_message}")
            return render(request, 'main/payment.html', {'plan': plan})

    return render(request, 'main/payment.html', {'plan': plan})
def calculate_price(plan):
    plan = plan.lower()

    if plan in ['standard_pro', 'standard pro']:
        return 7000
    elif plan in ['expert_farm', 'expert farm']:
        return 15000
    elif plan in ['investor_smes', 'investor/smes']:
        return 30000
    elif plan in ['trial_pack', 'trial pack']:
        return 3000
    return 10000  # default fallback

def create_checkout_session(request):
    if request.method == "POST":
        plan = request.POST.get("plan")  # This comes from the form radio button
        if not plan:
            messages.error(request, "Please select a plan.")
            return redirect("pricing")

        request.session["selected_plan"] = plan  # ✅ Store plan in session

        # Create Stripe checkout session here...
        session = stripe.checkout.Session.create(
            # ...
            success_url=request.build_absolute_uri(reverse("signup")),
            cancel_url=request.build_absolute_uri(reverse("pricing")),
            # ...
        )

        return redirect(session.url, code=303)

