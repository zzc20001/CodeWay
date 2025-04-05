import { createRootRoute, Outlet } from '@tanstack/react-router'
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools'
import { SidebarProvider } from '@/components/ui/sidebar'

export const Route = createRootRoute({
  component: () => (
    <div className="flex h-screen w-full overflow-hidden">
      <SidebarProvider defaultOpen={true}>
        <Outlet />
      </SidebarProvider>
      {process.env.NODE_ENV === 'development' && <TanStackRouterDevtools />}
    </div>
  ),
})