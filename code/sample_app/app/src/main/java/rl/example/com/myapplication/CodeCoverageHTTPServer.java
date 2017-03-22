package rl.example.com.myapplication;

import android.net.wifi.WifiManager;
import android.util.Log;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

import fi.iki.elonen.NanoHTTPD;

/**
 * Created by Venelin Valkov <venelin@curiousily.com>
 * on 3/22/17.
 */
class CodeCoverageHTTPServer extends NanoHTTPD {

    private static final int PORT = 8981;
    private static final String TAG = CodeCoverageHTTPServer.class.getSimpleName();

    private final java.lang.reflect.Method executionDataMethod;
    private final Object jacocoAgent;

    public CodeCoverageHTTPServer() throws Exception {
        super(PORT);
        Class<?> jacocoRTClass = Class.forName("org.jacoco.agent.rt.RT");
        java.lang.reflect.Method instanceMethod = jacocoRTClass.getMethod("getAgent");
        jacocoAgent = instanceMethod.invoke(null);
        executionDataMethod = jacocoAgent.getClass().getMethod("getExecutionData", boolean.class);
    }

    @Override
    public Response serve(IHTTPSession session) {
        try {
            byte[] execData = (byte[]) executionDataMethod.invoke(jacocoAgent, false);
            InputStream is = new ByteArrayInputStream(execData);
            return newFixedLengthResponse(Response.Status.OK, "application/octet-stream", is, execData.length);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void start(WifiManager wifiManager) {
        try {
            CodeCoverageHTTPServer server = new CodeCoverageHTTPServer();
            server.start();
            int ipAddress = wifiManager.getConnectionInfo().getIpAddress();
            final String serverIpAddress = String.format("%d.%d.%d.%d",
                    (ipAddress & 0xff),
                    (ipAddress >> 8 & 0xff),
                    (ipAddress >> 16 & 0xff),
                    (ipAddress >> 24 & 0xff));
            Log.d(TAG, "http://" + serverIpAddress + ":" + PORT);
        } catch (Exception e) {
            throw new RuntimeException("Server not started. Unable to find Jacoco agent", e);
        }
    }
}