# Per-repo fleet start config for stems-mcp
# Edit ports/backend target here - start.ps1 is fleet-standard.
@{
    Name         = 'stems-mcp'
    BackendPort  = 11126
    FrontendPort = 11127
    HealthPath   = '/api/health'
    WebRoot      = '.'
    Backend = @{
        Kind          = 'uvicorn'
        UvicornTarget = 'stems_mcp.server:app'
        SyncExtras    = @('dev')
        Env           = @{ WEB_PORT = '11126' }
    }
    Frontend = @{
        Kind           = 'vite-npm'
        PackageManager = 'npm'
        PortEnvVar     = 'VITE_PORT'
        ApiTargetEnv   = 'VITE_API_TARGET'
    }
}
