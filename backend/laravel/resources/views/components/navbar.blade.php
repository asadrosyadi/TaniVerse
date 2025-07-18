
<nav
    class="layout-navbar container-xxl navbar navbar-expand-xl navbar-detached align-items-center bg-navbar-theme"
    id="layout-navbar" style="z-index: 100 !important">

    <div class="layout-menu-toggle navbar-nav align-items-xl-center me-3 me-xl-0 d-xl-none">
        <a class="nav-item nav-link px-0 me-xl-4" href="javascript:void(0)">
            <i class="bx bx-menu bx-sm"></i>
        </a>
    </div>

    <div class="navbar-nav-right d-flex align-items-center" id="navbar-collapse">
        <ul class="navbar-nav flex-row align-items-center ms-auto">
            <!-- Place this tag where you want the button to render. -->
           <!-- Language -->
            <li class="nav-item navbar-dropdown dropdown-user dropdown">
                <a class="nav-link dropdown-toggle hide-arrow" title="Language" href="javascript:void(0);" data-bs-toggle="dropdown">
                    <i class="bx bx-globe bx-sm"></i>
                </a>
                <ul class="dropdown-menu dropdown-menu-end" data-bs-popper="static">
                    @foreach ($languages as $code => $language)
                    <li>
                        <a class="dropdown-item {{ auth()->user()->locale === $code ? 'active' : '' }}" href="{{ route('account.locale') . '?locale=' . $code }}" data-language="{{ $code }}" data-text-direction="ltr">
                            <span class="align-middle">{{ $language }}</span>
                        </a>
                    </li>
                    @endforeach
                </ul>
            </li>
            <!--/ Language -->
            <!-- User -->
            <li class="nav-item navbar-dropdown dropdown-user dropdown">
                <a class="nav-link dropdown-toggle hide-arrow" href="javascript:void(0);" data-bs-toggle="dropdown">
                    <div class="avatar avatar-online">
                        <img src="{{ asset('img/profile/' . $user->image) }}" class="card-img">
                    </div>
                </a>
                <ul class="dropdown-menu dropdown-menu-end">
                    <li>
                        <a class="dropdown-item" href="#">
                            <div class="d-flex">
                                <div class="flex-shrink-0 me-3">
                                    <div class="avatar avatar-online">
                                        <img src="{{ asset('img/profile/' . $user->image) }}" class="card-img">
                                    </div>
                                </div>
                                <div class="flex-grow-1">
                                    <span class="fw-medium d-block">{{ auth()->user()->name }}</span>
                                    <small class="text-muted">-</small>
                                </div>
                            </div>
                        </a>
                    </li>
                    <li>
                        <div class="dropdown-divider"></div>
                    </li>
                    <li>
                        <a class="dropdown-item" href="{{ route('account.profile.edit') }}">
                            <i class="bx bx-user me-2"></i>
                            <span class="align-middle">{{ __('menu.account') }}</span>
                        </a>
                    </li>
                    <li>
                        <a class="dropdown-item" href="{{ route('account.log.index') }}">
                            <i class="bx bx-history me-2"></i>
                            <span class="align-middle">{{ __('menu.activity_log') }}</span>
                        </a>
                    </li>
                    <li>
                        <div class="dropdown-divider"></div>
                    </li>
                    <li>
                        <form action="{{ route('logout') }}" method="post" id="logout-menu-form-on-nav">@csrf</form>
                        <a class="dropdown-item" href="javascript:void(0);" id="logout-menu-button-on-nav">
                            <i class="bx bx-power-off me-2"></i>
                            <span class="align-middle">{{ __('auth.logout') }}</span>
                        </a>
                    </li>
                </ul>
            </li>
            <!--/ User -->
        </ul>
    </div>
</nav>
