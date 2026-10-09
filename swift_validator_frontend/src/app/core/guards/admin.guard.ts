import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { map, catchError, of } from 'rxjs';
import { AuthService } from '../services/auth.service';

export const adminGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);

  return auth.me().pipe(
    map(user => {
      if (user.role === 'ADMIN') return true;
      return router.createUrlTree(['/app']);
    }),
    catchError(() => of(router.createUrlTree(['/login'])))
  );
};
